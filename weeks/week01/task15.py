import numpy as np
import torch


def main() -> None:
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(
        f"Numpy array:\n {array} \n",
        f"dtype={array.dtype}, shape={array.shape}\n"
          )

    tensor = torch.from_numpy(array)
    print(
        f"PyTorch tensor:\n {tensor} \n",
        f"dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}\n"
        )

    device = torch_selected_device()
    print(f"Selected device: {device}")

    tensor = tensor.to(device)
    print(f"Tensor moved to device: {device}, dtype={tensor.dtype}")

def torch_selected_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")