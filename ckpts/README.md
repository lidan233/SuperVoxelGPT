# SuperVoxelGPT — checkpoints

Everything the text→shape pipeline loads. None of it is in git.

```bash
bash fetch_ckpts.sh          # downloads here and verifies against MD5SUMS.txt
```

The published folder is

  https://utdallas.box.com/s/a9aekucnkxkk4q8oon85gbfqbfmqejvv

and it asks for a login, so `curl` and `wget` fetch the sign-in page rather than
the weights. `fetch_ckpts.sh` goes through `rclone` against a remote you have
configured for that account; `REMOTE=...` overrides where it looks.

Downloaded by hand instead? Verify before using them — a truncated checkpoint
still loads, with missing keys, and only the numbers come out wrong:

```bash
md5sum -c MD5SUMS.txt
```

## Inference

Three stages. Each row is loaded at the stage shown.

| file | size | stage | what it is |
|---|---|---|---|
| `clipText/` | 471 M | ① | CLIP text tower (`openai/clip-vit-large-patch14`, revision `32bd6428`). Shipped rather than downloaded so the encoding does not drift with upstream. Must stay a directory — `CLIPTokenizer.from_pretrained` reads the standard filenames inside it. |
| `stage1Text2Saliency.bin` | 865 M | ① | text → 64³ occupancy + saliency. 24-layer bidirectional MaskGIT, 512 tokens, vocab 5625. Occupancy dice 0.9996. |
| `stage1SaliencyVaeEncoder.pt` | 270 M | ① | VQ-VAE encoder for the stage-① field |
| `stage1SaliencyVaeOccupancyDecoder.pt` | 677 M | ① | decodes occupancy |
| `stage1SaliencyVaeSaliencyDecoder.pt` | 677 M | ① | decodes saliency. **Keep this one in fp32** — bf16 cuDNN convolution is the source of cross-architecture drift. |
| `stage1SaliencyVaeConfig.json` | 3 K | ① | structure for the three above |
| `stage2TextSaliency2Shape.bin` | 791 M | AR | text → supervoxel tokens. Qwen AR, 12 layers, hidden 896, vocab 10125+2. |
| `stage2SupervoxelVaeEncoder.pt` | 1.4 G | ③ | SuperVoxel VAE encoder, FSQ vocab 10125 |
| `stage2SupervoxelVaeDecoder.pt` | 1.9 G | ③ | 64³ → 1024³ subdivision decoder |
| `stage2SupervoxelVaeConfig.json` | 5 K | ③ | structure; also names the two files below |

Stage ② loads no weights — it JIT-compiles a CUDA kernel.

## Training only

| file | size | what it is |
|---|---|---|
| `trellisEncoderOriginal.safetensors` | 709 M | the pretrained TRELLIS sparse-convolution backbone the SuperVoxel VAE was fine-tuned from |
| `trellisDecoderOriginal.safetensors` | 948 M | same, decoder side |

**Inference does not need these.** `stage2SupervoxelVaeConfig.json` names them as
`models.encoder.weights` and `models.decoder.weights`, so they are loaded when the model
is constructed — and then every one of their parameters is overwritten by
`stage2SupervoxelVae{Encoder,Decoder}.pt` on the next line. Measured: 284/284 encoder
parameters and 292/292 decoder parameters are replaced. The released `.pt` files are
already written in the model's own namespace, so inference loads them straight, with no
remapping. Nothing of the backbone survives into the weights that actually run.

Training is where they matter: `train/stage2TextSaliency2Shape/train_vae.py` starts from
them, which is the lineage the released autoencoder has. The config names them by bare
filename, resolved against this directory (or `--ckpts`). If they are absent the run warns
and starts from random initialisation instead — it will train, but not to the released model.

