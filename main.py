import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from model.model_v2 import SteganoModel

import torchvision.transforms.v2 as v2
from torchvision.transforms.v2.functional import to_pil_image

# model_dir = 'Model_v2/stegano_model_final.pth'
model_dir = 'Model_v4/stegano_model_final.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(model_dir):
    print(f"Trained model file '{model_dir}' not found!")
    raise FileNotFoundError(f"Trained model file '{model_dir}' not found!")

model = SteganoModel()
model.load_state_dict(torch.load(model_dir, map_location=device))
model.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def save_tensor_as_image(tensor, save_path):
    tensor_image = torch.clamp(tensor, 0, 1)
    image = to_pil_image(tensor_image.squeeze(0).cpu())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)    
    image.save(save_path)
    print(f"Image saved to {save_path}")
    
def encode(cover_path, secret_path, stego_path):
    cover_image = Image.open(cover_path).convert("RGB")
    secret_image = Image.open(secret_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)
    secret_tensor = transform(secret_image).unsqueeze(0).to(device)

    _, _, cover_h, cover_w = cover_tensor.shape
    _, _, secret_h, secret_w = secret_tensor.shape

    pad_h_size = 0
    pad_w_size = 0
    if cover_h < secret_h or cover_w < secret_w:
        raise ValueError("Cover image must be larger than secret image.")
    if cover_h > secret_h or cover_w > secret_w:
        pad_h_size = cover_h - secret_h
        pad_w_size = cover_w - secret_w
    if pad_h_size > 0 or pad_w_size > 0:
        secret_tensor = F.pad(secret_tensor, (0, pad_w_size, 0, pad_h_size), mode='constant', value=0)

    with torch.no_grad():
        stego_tensor, _ = model(cover_tensor, secret_tensor)

    save_tensor_as_image(stego_tensor, stego_path)

def decode(stego_path, reveal_path):
    stego_image = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)

    with torch.no_grad():
        revealed_tensor = model.secret_extractor(stego_tensor)

    save_tensor_as_image(revealed_tensor, reveal_path)

if __name__ == "__main__":
    # main()
    cover_path = "test/1024x1024_Cover.jpg"  
    secret_path = "test/1024x1024_Secret.jpg"  

    stego_path = "output/stego_image.png"  
    reveal_path = "output/reveal_image.png"  

    if stego_path.endswith('.jpg'):
        stego_path = stego_path.replace('.jpg', '.png')
    if reveal_path.endswith('.jpg'):
        reveal_path = reveal_path.replace('.jpg', '.png')

    encode(cover_path, secret_path, stego_path)
    decode(stego_path, reveal_path)