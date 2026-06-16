"""
Smoke test: does bitsandbytes 4-bit quantization work on Intel XPU?

Run inside the training container:
  docker compose run --rm trainer python /app/scripts/smoke_bnb_xpu.py

Pass  → proceed with the custom NF4 expert path
Fail  → check bitsandbytes XPU backend compilation
"""
import sys
import torch

print(f"Python  {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
if not torch.xpu.is_available():
    print("FAIL: XPU not available")
    sys.exit(1)

print(f"XPU device: {torch.xpu.get_device_name(0)}")
print(f"XPU VRAM:   {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
print(f"\nbitsandbytes {bnb.__version__}")

PASS = True

# ── Test 1: quantize_4bit on CPU ─────────────────────────────────────────────
print("\n[1] quantize_4bit on CPU (gate_up_proj-sized slice)")
w_cpu = torch.randn(1408, 2816, dtype=torch.bfloat16)
qw, qs = quantize_4bit(w_cpu, quant_type="nf4", blocksize=64,
                        compress_statistics=False, quant_storage=torch.uint8)
print(f"    w_cpu: {w_cpu.shape} {w_cpu.dtype} {w_cpu.nbytes/1024**2:.1f} MiB")
print(f"    qw:    {qw.shape}  {qw.dtype}  {qw.nbytes/1024**2:.1f} MiB  (expected ~½)")
assert qw.dtype == torch.uint8, f"FAIL: expected uint8, got {qw.dtype}"
print("    PASS")

# ── Test 2: dequantize_4bit on CPU ───────────────────────────────────────────
print("\n[2] dequantize_4bit on CPU (round-trip error)")
dw_cpu = dequantize_4bit(qw, qs).to(torch.bfloat16)
err = (w_cpu - dw_cpu).abs().mean().item()
print(f"    mean abs error: {err:.5f}  (expected ≈ 0.01–0.05 for NF4)")
assert err < 0.5, f"FAIL: reconstruction error too large ({err})"
print("    PASS")

# ── Test 3: move quantized tensors to XPU ────────────────────────────────────
print("\n[3] move qw + quant_state to XPU")
try:
    qw_xpu = qw.to("xpu")
    qs_xpu = qs.to("xpu")
    torch.xpu.synchronize()
    print(f"    qw on XPU: {qw_xpu.device}  {qw_xpu.dtype}  {qw_xpu.nbytes/1024**2:.1f} MiB")
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    PASS = False
    qw_xpu = qw_cpu = None

# ── Test 4: dequantize_4bit on XPU ───────────────────────────────────────────
if qw_xpu is not None:
    print("\n[4] dequantize_4bit cross-device: qw on XPU, qs on CPU")
    try:
        # _NF4Experts3D pattern: packed weights on XPU, QuantState stays on CPU.
        # QuantState.to() returns None in bnb 0.49 (mutates in-place) so we never
        # call it.  dequantize_4bit handles the cross-device case internally.
        dw_xpu = dequantize_4bit(qw_xpu, qs).to(torch.bfloat16)
        torch.xpu.synchronize()
        # Ensure result is on XPU (bnb may return CPU tensor)
        target = qw_xpu.device
        if dw_xpu.device != target:
            dw_xpu = dw_xpu.to(target)
        err_xd = (dw_cpu - dw_xpu.cpu()).abs().mean().item()
        print(f"    result: {dw_xpu.shape}  {dw_xpu.dtype}  on {dw_xpu.device}")
        print(f"    round-trip error (same as CPU dequant): {err_xd:.6f}  (expected ≈ 0.0)")
        assert err_xd < 1e-4, f"FAIL: cross-device dequant mismatch ({err_xd})"
        print("    PASS")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    FAIL: {e}")
        PASS = False

# ── Test 5: Linear4bit forward on XPU ────────────────────────────────────────
print("\n[5] bnb.nn.Linear4bit forward on XPU (attention substitute)")
try:
    lin = bnb.nn.Linear4bit(2816, 2816, bias=False, quant_type="nf4",
                             compute_dtype=torch.bfloat16).to("xpu")
    x = torch.randn(4, 2816, device="xpu", dtype=torch.bfloat16)
    out = lin(x)
    torch.xpu.synchronize()
    print(f"    out: {out.shape}  {out.dtype}  on {out.device}")
    assert out.shape == (4, 2816), f"FAIL: unexpected shape {out.shape}"
    assert not out.isnan().any(), "FAIL: NaN in output"
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    PASS = False

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if PASS:
    print("ALL TESTS PASSED — proceed with custom NF4 expert path")
    sys.exit(0)
else:
    print("SOME TESTS FAILED — review output above before proceeding")
    sys.exit(1)
