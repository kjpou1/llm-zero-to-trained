import torch


def get_device_info():
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"


def print_torch_info():
    print("=" * 40)
    print("🔍 PyTorch Environment Info")
    print("-" * 40)
    print(f"{'PyTorch version:':20} {torch.__version__}")
    print(f"{'Available device:':20} {get_device_info()}")
    print("=" * 40)


if __name__ == "__main__":
    print_torch_info()
