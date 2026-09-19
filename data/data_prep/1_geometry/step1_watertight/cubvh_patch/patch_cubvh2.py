"""Widen cubvh's floodfill cell indexing so it survives grids above 2^31 cells.

Purpose
    Stock cubvh indexes flood cells with int32, which overflows past roughly 1290^3 voxels. This
    rewrites the affected CUDA source in place so high-resolution extraction stays correct.

Input
    A cloned cubvh source tree. The script git-checkouts the target file first, so it is idempotent.

Output
    The same tree with its floodfill header rewritten; rebuild the package afterwards.

Key idea
    Labels are only ever compared for equality, never ordered or arithmetically combined. That means
    the storage type can stay int32 (4 bytes) and merely be reinterpreted as unsigned inside the
    kernel — memory usage does not grow at all. The unlabeled sentinel moves from -1 to UINT_MAX and
    every ">= 0" test becomes "!= UINT_MAX". Regex matching is used rather than exact strings because
    the upstream file carries significant trailing whitespace.
"""
import os, re, subprocess
from pathlib import Path


def main():
    W = Path(os.environ.get("CUBVH_SRC", "."))   # a cloned cubvh source tree
    rel = "include/gpu/floodfill.cuh"
    subprocess.run(["git", "-C", str(W), "checkout", "--", rel], check=True)
    F = W / rel
    s = F.read_text()
    print(f"restored {rel} ({len(s)} bytes)")

    R = []  # (regex, replacement, description)
    R += [(r"#include <cstdint>", "#include <cstdint>\n#include <climits>", "add climits")]
    # --- kernel signatures: int* labels -> unsigned int* labels; int Ntot/Nvol -> size_t
    R += [(r"__global__ void initLabels\(const bool\*\s*__restrict__\s*grid,\s*int\s*\*\s*__restrict__\s*labels,\s*int\s+Ntot\)",
           "__global__ void initLabels(const bool* __restrict__ grid, unsigned int * __restrict__ labels, size_t Ntot)", "initLabels signature")]
    R += [(r"__global__ void compress\(int\s*\*\s*__restrict__\s*labels,\s*int\s+Ntot\)",
           "__global__ void compress(unsigned int * __restrict__ labels, size_t Ntot)", "compress signature")]
    R += [(r"int\*\s*__restrict__\s*labels,\s*\n(\s*)int\*\s*__restrict__\s*changed,",
           r"unsigned int* __restrict__ labels,\n\1      int*  __restrict__ changed,", "hookBatch signature")]
    R += [(r"int Nvol,(\s*)//\s*stride between consecutive batches", r"size_t Nvol,\1// stride", "hookBatch Nvol")]
    R += [(r"int Ntot\)(\s*)//\s*total number of voxels", r"size_t Ntot)\1// total", "hookBatch Ntot")]
    # --- thread index: int idx = blockIdx.x*blockDim.x+threadIdx.x -> size_t
    R += [(r"int idx = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;",
           "size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;", "thread index -> size_t")]
    # --- the sentinel assignment inside initLabels
    R += [(r"labels\[idx\] = grid\[idx\] \? -1 : idx;",
           "labels[idx] = grid[idx] ? UINT_MAX : (unsigned int)idx;", "sentinel -1 -> UINT_MAX")]
    # --- locals inside hookBatch
    R += [(r"int batch_idx = idx / Nvol;", "size_t batch_idx = idx / Nvol;", "batch_idx")]
    R += [(r"int local = idx % Nvol;", "size_t local = idx % Nvol;", "local")]
    R += [(r"int x =\s*local % W;", "long long x = (long long)(local % W);", "x")]
    R += [(r"int y = \(local / W\) % H;", "long long y = (long long)((local / W) % H);", "y")]
    R += [(r"int z =\s*local / \(W \* H\);", "long long z = (long long)(local / ((size_t)W * H));", "z")]
    R += [(r"int batch_base = batch_idx \* Nvol;", "size_t batch_base = batch_idx * Nvol;", "batch_base")]
    R += [(r"int best = labels\[idx\];", "unsigned int best = labels[idx];", "best")]
    # --- neighbours: int n = idx +/- k -> size_t; labels[n] >= 0 -> != UINT_MAX
    R += [(r"int n = idx ([-+]) (1|W|W \* H);", r"size_t n = idx \1 (size_t)(\2);", "neighbour index")]
    R += [(r"labels\[n\] >= 0", "labels[n] != UINT_MAX", "neighbour validity")]
    # --- atomic
    R += [(r"int current_label = labels\[idx\];", "unsigned int current_label = labels[idx];", "current_label")]
    R += [(r"if \(current_label >= 0 && best < current_label\)", "if (current_label != UINT_MAX && best < current_label)", "atomic condition")]
    R += [(r"int old_val = atomicCAS", "unsigned int old_val = atomicCAS", "atomicCAS")]
    # --- inside compress
    R += [(r"if \(idx >= Ntot \|\| labels\[idx\] < 0\) return;", "if (idx >= Ntot || labels[idx] == UINT_MAX) return;", "compress skip")]
    R += [(r"int current = idx;", "size_t current = idx;", "compress current")]
    R += [(r"int root = labels\[current\];", "size_t root = (size_t)labels[current];", "compress root")]
    R += [(r"while \(root != current && root >= 0 && root < Ntot && steps < MAX_STEPS\)", "while (root != current && root < Ntot && steps < MAX_STEPS)", "compress while1")]
    R += [(r"root = labels\[current\];", "root = (size_t)labels[current];", "compress root update")]
    R += [(r"labels\[idx\] = idx;\n(\s*)return;", r"labels[idx] = (unsigned int)idx;\n\1return;", "compress self-reference")]
    R += [(r"while \(current != root && current >= 0 && current < Ntot && steps < MAX_STEPS\)", "while (current != root && current < Ntot && steps < MAX_STEPS)", "compress while2")]
    R += [(r"int next = labels\[current\];", "size_t next = (size_t)labels[current];", "compress next")]
    R += [(r"labels\[current\] = root;", "labels[current] = (unsigned int)root;", "compress path compression")]
    # --- host
    R += [(r"static int divUp\(int a, int b\)", "static size_t divUp(size_t a, size_t b)", "divUp")]
    R += [(r"const int Nvol = H \* W \* D;", "const size_t Nvol = (size_t)H * W * D;", "host Nvol")]
    R += [(r"const int Ntot = B \* Nvol;", "const size_t Ntot = (size_t)B * Nvol;", "host Ntot")]
    R += [(r"const size_t bytesLabel = Ntot \* sizeof\(int\);", "const size_t bytesLabel = Ntot * sizeof(unsigned int);", "bytesLabel")]
    R += [(r"int\s+\*d_labels\s*=\s*nullptr;", "unsigned int *d_labels = nullptr;", "d_labels type")]
    R += [(r"int blocks = divUp\(Ntot, THREADS_PER_BLOCK\);", "size_t blocks = divUp(Ntot, THREADS_PER_BLOCK);", "blocks")]
    R += [(r"<<<blocks, THREADS_PER_BLOCK>>>", "<<<(unsigned int)blocks, THREADS_PER_BLOCK>>>", "kernel launch")]
    R += [(r"cudaMemcpy\(mask, d_labels, bytesLabel, cudaMemcpyDeviceToHost\);",
           "cudaMemcpy(mask, d_labels, bytesLabel, cudaMemcpyDeviceToHost);  // bit-pattern copy: unsigned -> int32 storage", "copy back")]

    tot = 0
    for pat, rep, why in R:
        s2, n = re.subn(pat, rep, s)
        if n == 0:
            print(f"  [x] no match: {why}")
        else:
            tot += n; s = s2
            print(f"  [ok] {why}  ({n} sites)")
    F.write_text(s)
    print(f"\nreplaced {tot} sites -> {F}")
    print("checking for leftover 'int Ntot' / 'int idx':")
    for bad in ("int Ntot", "int idx = blockIdx", "int * __restrict__ labels", "labels[idx] = grid[idx] ? -1"):
        print(f"  '{bad}': {s.count(bad)} occurrences")


if __name__ == "__main__":
    main()
