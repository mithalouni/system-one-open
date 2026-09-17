"""Shared Modal image/volume definitions."""
import modal, os
vol = modal.Volume.from_name("jev-replica", create_if_missing=True)
base_image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==5.17.0", "accelerate==1.15.0", "peft==0.21.0", "hf_transfer",
                         "sentencepiece", "protobuf", "numpy", "fastapi[standard]")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}))
image = base_image.add_local_python_source("s1")


def gpu_kwargs(default="H100"):
    g = os.environ.get("S1_GPU", default)
    if g.lower() == "none":
        return dict(cpu=float(os.environ.get("S1_CPU", "8")), memory=int(os.environ.get("S1_MEM", "32768")))
    return dict(gpu=g)
