"""Run a See-through inference script on GPUs without native bf16 (T4, V100: compute capability < 8.0).

See-through loads every model in bf16. On these GPUs the elementwise bf16 ops work, but cuBLAS
has no bf16 GEMM, so the first matmul fails with CUBLAS_STATUS_ALLOC_FAILED / NOT_INITIALIZED.
Without editing See-through, this shim:
  1. runs the whole script under torch.autocast(cuda, float16): every matmul/conv (including the
     bitsandbytes NF4 linears, which call F.linear internally) executes in fp16 instead of bf16;
  2. keeps the VAEs in fp32 with autocast disabled inside them, because the SDXL VAE overflows
     to NaN in fp16. See-through already casts VAE inputs to `vae.dtype`, so fp32 is accepted.

usage: python -m pipeline.t4_shim <see-through script> [script args...]   (run from the see-through repo root)
"""
import functools
import os
import runpy
import sys

import torch

FP16 = torch.float16


def _to_fp32(x):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.float()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_fp32(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_fp32(v) for k, v in x.items()}
    return x


def _fp32_method(fn):
    """Run fn with autocast off and floating tensor args in fp32."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # some VAEs are loaded with torch_dtype=bf16 and only moved to cuda later (Marigold NF4)
        p = next(self.parameters(), None)
        if p is not None and p.dtype != torch.float32:
            torch.nn.Module.to(self, torch.float32)
        with torch.autocast("cuda", enabled=False):
            return fn(self, *_to_fp32(args), **_to_fp32(kwargs))
    return wrapper


def _pin_fp32_to(cls):
    """Module.to(dtype=bf16/fp16) on a VAE becomes fp32; device moves are untouched."""
    orig = cls.to

    @functools.wraps(orig)
    def to(self, *args, **kwargs):
        if kwargs.get("dtype") in (torch.bfloat16, torch.float16):
            kwargs["dtype"] = torch.float32
        args = tuple(torch.float32 if a in (torch.bfloat16, torch.float16) else a for a in args)
        return orig(self, *args, **kwargs)
    cls.to = to


def patch_vaes():
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    _pin_fp32_to(AutoencoderKL)
    for name in ("encode", "decode", "_encode", "_decode", "forward"):
        if hasattr(AutoencoderKL, name):
            setattr(AutoencoderKL, name, _fp32_method(getattr(AutoencoderKL, name)))
    try:
        from modules.layerdiffuse import vae as lvae
    except ImportError:
        return
    _pin_fp32_to(lvae.TransparentVAE)
    for cls, names in [(lvae.TransparentVAEDecoder, ("forward", "estimate_single_pass", "estimate_augmented")),
                       (lvae.TransparentVAEEncoder, ("forward", "encode"))]:
        for name in names:
            if hasattr(cls, name):
                setattr(cls, name, _fp32_method(getattr(cls, name)))



def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    # the script does sys.path.append(<repo>/inference); modules come from the installed common pkg
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(script))))
    patch_vaes()
    print("[t4_shim] bf16 GEMMs -> fp16 autocast, VAEs in fp32", flush=True)
    with torch.autocast("cuda", dtype=FP16):
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
