import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import numpy as np

def calculate_capacity_tensor(secret_array, reveal_array):
    if secret_array.shape != reveal_array.shape:
        raise ValueError("Secret and reveal images must have the same dimensions.")
    
    sum = 0
    abs_diff = torch.abs(secret_array - reveal_array)
    mean_diff = torch.mean(abs_diff)

    decoded_rate = 1 - mean_diff.item()
    print(f"Decoded Rate: {decoded_rate:.4f}")

    capacity = decoded_rate * 24
    print(f"Capacity: {capacity:.4f} bits per pixel")
    
    return decoded_rate, capacity

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def calculate_capacity(secret_array, reveal_array):
    if secret_array.shape != reveal_array.shape:
        raise ValueError("Secret and reveal images must have the same dimensions.")
    
    sum = 0
    abs_diff = np.abs(secret_array - reveal_array)
    mean_diff = np.mean(abs_diff)

    decoded_rate = 1 - mean_diff
    print(f"Decoded Rate: {decoded_rate:.4f}")

    capacity = decoded_rate * 24
    print(f"Capacity: {capacity:.4f} bits per pixel")
    
    return decoded_rate, capacity

if __name__ == "__main__":
    secret_path = "test/Secret_Square.jpg"
    reveal_path = "dist/reveal_image.png"

    secret_image = Image.open(secret_path).convert("RGB")
    reveal_image = Image.open(reveal_path).convert("RGB")

    # secret_array = transform(secret_image)
    # reveal_array = transform(reveal_image)

    secret_array = np.array(secret_image).astype(np.float32) / 255.0
    reveal_array = np.array(reveal_image).astype(np.float32) / 255.0

    print(f"Secret Image Shape: {secret_array.shape}")
    print(f"Reveal Image Shape: {reveal_array.shape}")

    decoded_rate, capacity = calculate_capacity(secret_array, reveal_array)