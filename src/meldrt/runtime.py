"""Process-level setup shared by the scripts and the live app."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def expose_torch_cuda_dlls() -> None:
    """faster-whisper (CTranslate2) and llama.cpp need the cuBLAS and cuDNN DLLs. PyTorch's
    Windows wheel already ships them in torch/lib, so point the DLL search there instead of
    installing the CUDA toolkit separately."""
    if sys.platform == "win32":
        torch_lib = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(torch_lib)
