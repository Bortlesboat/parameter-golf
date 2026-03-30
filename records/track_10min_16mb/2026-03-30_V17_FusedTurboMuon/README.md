# Record: Turbo-Muon + EngramLite + ParamBanking + GPTQ Reserve Optimization (val_bpb 1.1128)

**val_bpb: 1.1128** (3-seed mean, std 0.0008) | **~15.98 MB** | 8xH100 SXM, 600s train, ~120s eval

Built on [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @mikeapedia, with fused Triton MLP kernel architecture from [PR #1072](https://github.com/openai/parameter-golf/pull/1072) by @vimeto.

## Results (8xH100 SXM, SWA applied)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | **1.1131** | 15,978,243 |
| 42 | **1.1119** | 15,988,373 |
| 999 | **1.1133** | 15,983,867 |
| **Mean +/- Std** | **1.1128 +/- 0.0008** | |

## What's New vs PR #1089

### 1. GPTQ Reserve Optimization
Reduced GPTQ calibration reserve from 14s to 9s. Actual calibration takes ~8.4s across all tested runs, so 14s wastes 5+ seconds of training budget. Recovers ~55 extra training steps at 107ms/step.

### 2. Fused Triton MLP Kernel (Forward-Only Architecture)
Integrated PR #1072's fused Triton kernel for `matmul + LeakyReLU(0.3) + square` — eliminates intermediate HBM writes. Key design: forward-only fusion with standard PyTorch backward, based on PR #1105's finding that Triton backward causes eager mode (2.7x slower). Registered via `torch._dynamo.allow_in_graph()`.

**Status:** Kernel is included but disabled in this run due to `torch.compile(fullgraph=True)` incompatibility with custom `autograd.Function` + Triton TensorDescriptor API on PyTorch 2.9. Falls back to standard MLP path gracefully. Future work: port to `torch.library.custom_op` API for full compile compatibility.

### 3. Centralized Activation Slope
All `negative_slope` references unified via `_NEGATIVE_SLOPE = 0.3` constant. Backward derivative uses derived `_SLOPE_SQ = _NEGATIVE_SLOPE ** 2` to stay in sync.

## Architecture (from PR #1089)

- 11L, 512d, 8H/4KV (GQA), MLP 3.5x LeakyReLU(0.3)^2
- Turbo-Muon optimizer (AOL + Polar Express + row_col norm, 4 NS iterations)
- EngramLite (bigram + trigram, 2 heads, 8192 buckets)
- Parameter Banking (3D bank tensors, batched Newton-Schulz)
- U-Net sigmoid-gated skip connections + ValueEmbedding
- SmearGate, Partial RoPE(16), LN Scale
- SWA(0.2 threshold, every 50 steps) + EMA(0.997) fallback
- Mixed-precision GPTQ int5/int6/int7 (Hessian sensitivity)
- Brotli + byte-shuffle compression
- F.scaled_dot_product_attention (SDPA, auto-selects FA3 backend)

## Timing

| Phase | Time |
|-------|------|
| Training (~5,632 steps @ 107ms) | 591s |
| GPTQ calibration + quantization | 9s (reserved) |
| Sliding window eval (stride=64) | ~120s |

## Reproduction

```bash
# Use official template: runpod/parameter-golf:latest (PyTorch 2.9.1+cu128)
pip install brotli
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

GPTQ_RESERVE_MS=9000 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Rule Compliance

- Standard F.cross_entropy scoring (softmax, sum=1)
- No eval-time training data access
- Artifact < 16,000,000 bytes (all 3 seeds)
- Training < 600s, eval < 600s
- Causal sliding-window evaluation (stride=64)

## Credits

- **Turbo-Muon + EngramLite + ParamBanking**: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @mikeapedia
- **Fused Triton MLP kernel**: [PR #1072](https://github.com/openai/parameter-golf/pull/1072) by @vimeto
- **Forward-only fusion insight**: [PR #1105](https://github.com/openai/parameter-golf/pull/1105) by @abaybektursun
- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun
