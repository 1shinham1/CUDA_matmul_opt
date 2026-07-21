# Rebuilding flash_attn_cuda.so

`flash_attn_cuda.cpython-310-x86_64-linux-gnu.so` (the compiled extension
`import flash_attn` actually loads) is gitignored (`*.so`, standard practice
for build artifacts) and `csrc/flash_attn/cutlass/` was deleted after the
build to save space (88MB, third-party dependency, only needed at compile
time). This is how to reproduce both from a fresh clone.

This repo (`FA_official/`) is FlashAttention **v1.0.9** (FA1, not the FA2/FA3
that `pip install flash-attn` installs today) -- Ada (RTX 4090, sm_89) isn't
one of upstream's build targets since this repo predates Ada, so `setup.py`
here has one deliberate diff from upstream: an added `-gencode
arch=compute_89,code=sm_89` (see the comment in `setup.py` around line 121).
Without it the extension has no cubin/PTX this GPU can run.

## 1. Matching CUDA toolkit

`torch.utils.cpp_extension` hard-fails if `nvcc --version`'s major version
doesn't match `torch.version.cuda`'s major version. Check what your torch
was built against:

```
python -c "import torch; print(torch.version.cuda)"
```

This project's `cuda_env` conda env has torch 2.11+cu128 (CUDA 12.8), while
the system-wide `/usr/local/cuda` is 13.1 -- a major-version mismatch that
fails the check. Fix: install a matching toolkit *inside* the conda env
instead of touching the system one:

```
conda install -n cuda_env -c nvidia cuda-toolkit=12.8.1 -y
```

Conda's nvidia-channel toolkit puts headers/libs under
`targets/x86_64-linux/{include,lib}` rather than the conventional
`include/lib`, so host (non-nvcc) compilation steps won't find
`cuda_runtime.h` unless you point `CPATH`/`LIBRARY_PATH` there explicitly:

```
export CUDA_HOME=/home/shshin/miniconda3/envs/cuda_env
export CPATH=$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## 2. Vendor cutlass at the exact pinned commit

`setup.py` normally does `git submodule update --init csrc/flash_attn/cutlass`,
but that only works if this checkout is a real git submodule of a parent
repo (it isn't here -- `FA_official/` was vendored as plain files). Fetch it
manually at the commit flash-attention v1.0.9 actually pins (confirmed via
`https://api.github.com/repos/Dao-AILab/flash-attention/contents/csrc/flash_attn/cutlass?ref=v1.0.9`):

```
cd FlashAttention-implementation/official
git clone https://github.com/NVIDIA/cutlass.git csrc/flash_attn/cutlass_tmp
cd csrc/flash_attn/cutlass_tmp
git checkout 319a389f42b776fae5701afcb943fc03be5b5c25
cd ../../..
rmdir csrc/flash_attn/cutlass  # empty placeholder
mv csrc/flash_attn/cutlass_tmp csrc/flash_attn/cutlass
rm -rf csrc/flash_attn/cutlass/.git
```

(`setup.py` already skips the `git submodule update` call when it finds
`csrc/flash_attn/cutlass` non-empty, so this is enough.)

## 3. Build

```
pip install ninja packaging  # ninja: parallel build, ~5min instead of ~2h
cd FlashAttention-implementation/official
MAX_JOBS=8 pip install -e . --no-build-isolation
```

`-e` (editable) means the `.so` is built in place inside `FA_official/` and
`import flash_attn` in the `cuda_env` env points straight at this directory
-- nothing is copied into site-packages. `MAX_JOBS=8` was fine for FA1 (only
~8 kernel `.cu` files); this is *not* safe advice for FA2 (`pip install
flash-attn` proper), which compiles ~70+ files including some that can OOM
a 125GB machine at `MAX_JOBS=8` -- use `MAX_JOBS=4` there if you ever go
down that path instead.

## 4. Verify

```
python -c "
import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
qkv = torch.randn(256, 3, 4, 64, device='cuda', dtype=torch.float16)
cu = torch.arange(0, 257, step=128, device='cuda', dtype=torch.int32)
out = flash_attn_unpadded_qkvpacked_func(qkv, cu, 128, dropout_p=0.0, causal=False)
print('OK', out.shape)
"
```

Once verified, you can delete `csrc/flash_attn/cutlass/` again if you want
to save the space -- it's only needed at compile time.
