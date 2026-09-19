// Soft-CVT CUDA kernel:
//  - seeds are bucketed by a host-side sort (CSR), so the kernel has no race
//  - ring search stops only once ((ring-1)*bin_w)^2 exceeds the K-th nearest distance
//  - top-K insertion sort in registers, ties by seed index; fixed-order softmax;
//    int64 fixed-point atomic accumulation
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define TOPK_NEIGHBORS 8
#define WSCALE (1<<20)

__global__ void accumulate(const float* __restrict__ cells, const long long* __restrict__ cell2,
                           const double* __restrict__ rho, int n_cells,
                           const float* __restrict__ seeds, int n_seeds,
                           int nb, float bin_w,
                           const int* __restrict__ bin_start,   // nb^3+1 CSR
                           const int* __restrict__ bin_items,   // n_seeds, sorted by (bin, seedidx)
                           float inv_tau,
                           unsigned long long* __restrict__ acc_m,
                           unsigned long long* __restrict__ acc_x) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    float px = cells[c*3+0], py = cells[c*3+1], pz = cells[c*3+2];
    int bx = min(nb-1, max(0, (int)(px / bin_w)));
    int by = min(nb-1, max(0, (int)(py / bin_w)));
    int bz = min(nb-1, max(0, (int)(pz / bin_w)));
    float bestd[TOPK_NEIGHBORS]; int besti[TOPK_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < TOPK_NEIGHBORS; k++) { bestd[k] = 3.4e38f; besti[k] = -1; }
    for (int ring = 0; ring < nb; ring++) {
        if (besti[TOPK_NEIGHBORS-1] >= 0) {
            float safe = (ring - 1) * bin_w;          // smallest possible seed-to-cell distance in this ring
            if (safe > 0.f && safe * safe > bestd[TOPK_NEIGHBORS-1]) break;
        }
        for (int dx = -ring; dx <= ring; dx++)
        for (int dy = -ring; dy <= ring; dy++)
        for (int dz = -ring; dz <= ring; dz++) {
            if (max(abs(dx),max(abs(dy),abs(dz))) != ring) continue;
            int gx = bx+dx, gy = by+dy, gz = bz+dz;
            if (gx<0||gy<0||gz<0||gx>=nb||gy>=nb||gz>=nb) continue;
            int b = (gx*nb + gy)*nb + gz;
            for (int t = bin_start[b]; t < bin_start[b+1]; t++) {
                int sidx = bin_items[t];
                float ddx = px-seeds[sidx*3+0], ddy = py-seeds[sidx*3+1], ddz = pz-seeds[sidx*3+2];
                float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                for (int k = 0; k < TOPK_NEIGHBORS; k++) {
                    bool better = (d2 < bestd[k]) || (d2 == bestd[k] && (unsigned)sidx < (unsigned)besti[k]);
                    if (better) {
                        for (int m = TOPK_NEIGHBORS-1; m > k; m--) { bestd[m]=bestd[m-1]; besti[m]=besti[m-1]; }
                        bestd[k] = d2; besti[k] = sidx; break;
                    }
                }
            }
        }
    }
    float dmin = bestd[0];
    float wsum = 0.f, w[TOPK_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < TOPK_NEIGHBORS; k++) {
        w[k] = (besti[k] >= 0) ? expf(-(bestd[k]-dmin) * inv_tau) : 0.f;
        wsum += w[k];
    }
    double r = rho[c];
    long long cx = cell2[c*3+0], cy = cell2[c*3+1], cz = cell2[c*3+2];
    #pragma unroll
    for (int k = 0; k < TOPK_NEIGHBORS; k++) {
        if (besti[k] < 0) continue;
        long long wi = (long long)((double)(w[k]/wsum) * r * (double)WSCALE + 0.5);
        if (wi <= 0) continue;
        atomicAdd(&acc_m[besti[k]], (unsigned long long)wi);
        atomicAdd(&acc_x[besti[k]*3+0], (unsigned long long)(wi * cx));
        atomicAdd(&acc_x[besti[k]*3+1], (unsigned long long)(wi * cy));
        atomicAdd(&acc_x[besti[k]*3+2], (unsigned long long)(wi * cz));
    }
}

torch::Tensor soft_cvt_cuda(torch::Tensor cells, torch::Tensor cell2, torch::Tensor rho,
                            torch::Tensor seeds0, double tau, int64_t iters, double bin_w_in) {
    auto seeds = seeds0.contiguous().clone();
    cells = cells.contiguous(); cell2 = cell2.contiguous(); rho = rho.contiguous();
    int n_cells = cells.size(0), n_seeds = seeds.size(0);
    int nb = (int)(256.0 / bin_w_in) + 1;
    float inv_tau = (float)(1.0/tau);
    auto opts_u = torch::TensorOptions().dtype(torch::kInt64).device(cells.device());
    auto acc_m = torch::zeros({n_seeds}, opts_u);
    auto acc_x = torch::zeros({n_seeds*3}, opts_u);
    for (int64_t it = 0; it < iters; it++) {
        // CSR bucketing; the sort key bin*n_seeds + idx breaks ties by index
        auto sx = seeds.select(1,0).clamp(0.0, 255.999f);
        auto sy = seeds.select(1,1).clamp(0.0, 255.999f);
        auto sz = seeds.select(1,2).clamp(0.0, 255.999f);
        auto bxi = (sx / bin_w_in).to(torch::kInt64).clamp(0, nb-1);
        auto byi = (sy / bin_w_in).to(torch::kInt64).clamp(0, nb-1);
        auto bzi = (sz / bin_w_in).to(torch::kInt64).clamp(0, nb-1);
        auto bid = (bxi * nb + byi) * nb + bzi;
        auto key = bid * (int64_t)n_seeds + torch::arange(n_seeds, opts_u);
        auto order = key.argsort();
        auto bin_items = order.to(torch::kInt32);
        auto sorted_bid = bid.index_select(0, order);
        auto bin_start = torch::searchsorted(sorted_bid,
            torch::arange(nb*(int64_t)nb*nb + 1, opts_u)).to(torch::kInt32);
        acc_m.zero_(); acc_x.zero_();
        auto strm = at::cuda::getCurrentCUDAStream();
        accumulate<<<(n_cells+127)/128, 128, 0, strm.stream()>>>(cells.data_ptr<float>(),
            (const long long*)cell2.data_ptr<int64_t>(),
            rho.data_ptr<double>(), n_cells, seeds.data_ptr<float>(), n_seeds, nb, (float)bin_w_in,
            bin_start.data_ptr<int>(), bin_items.data_ptr<int>(), inv_tau,
            (unsigned long long*)acc_m.data_ptr<int64_t>(), (unsigned long long*)acc_x.data_ptr<int64_t>());
        auto m = acc_m.clamp_min(1).to(torch::kFloat64);
        seeds = (acc_x.view({n_seeds,3}).to(torch::kFloat64) / m.unsqueeze(1) / 2.0).to(torch::kFloat32);
    }
    return seeds;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_cvt", &soft_cvt_cuda, "deterministic soft CVT v2");
}
