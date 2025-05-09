# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2.functional import to_pil_image
from model.model_v2 import SteganoModel
import text2image as tti

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Model_v4/stegano_model_final.pth"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = SteganoModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# Transform
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def save_image(tensor, save_path):
    tensor_image = torch.clamp(tensor, 0, 1)
    image = to_pil_image(tensor_image.squeeze(0).cpu()).convert("RGB")
    image.save(save_path)

# Encode function
def encode(cover_path, stego_path, secret_path):
    # Load cover image
    cover_img = Image.open(cover_path).convert("RGB")
    cover_tensor = transform(cover_img).unsqueeze(0).to(DEVICE)
    
    # Load or create secret
    if secret_path.endswith('.jpg') or secret_path.endswith('.png'):
        secret_img = Image.open(secret_path).convert("RGB")
        secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)
    elif secret_path.endswith('.txt'):
        cover_size = cover_img.size
        print(f"Cover size (width x height): {cover_size}")
        lines = tti.text_lines(secret_path)
        secret_img = tti.lines_image(lines, image_size=cover_size)
        secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)
    else:
        raise ValueError("Unsupported secret format")
    
    _, _, secret_h, secret_w = secret_tensor.shape
    _, _, cover_h, cover_w = cover_tensor.shape

    if secret_h > cover_h or secret_w > cover_w:
        raise ValueError("Secret image is larger than cover image")
    pad_h = 0
    pad_w = 0
    if cover_h > secret_h or cover_w > secret_w:
        pad_h = cover_h - secret_h
        pad_w = cover_w - secret_w
    if pad_h > 0 or pad_w > 0:
        secret_tensor = F.pad(secret_tensor, (0, pad_w, 0, pad_h), "constant", 0)
    
    # Forward pass
    with torch.no_grad():
        stego_tensor, _ = model(cover_tensor, secret_tensor)
    
    # Save stego_image
    if stego_path.endswith('.jpg'):
        stego_path = stego_path.replace('.jpg', '.png')
    if not stego_path.endswith('.png'):
        raise ValueError("Stego image must be saved as PNG")

    dir_name = os.path.dirname(stego_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_image(stego_tensor, stego_path)
    print(f"Encoded secret from {secret_path} into {stego_path}")

# Decode function
def decode(stego_path, output_path):
    # Load stego image
    stego_img = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_img).unsqueeze(0).to(DEVICE)
    
    # Forward pass with secret_extractor
    with torch.no_grad():
        reveal_tensor = model.secret_extractor(stego_tensor)
    
    if output_path.endswith('.jpg') or output_path.endswith('.png'):
        if output_path.endswith('.jpg'):
            output_path = output_path.replace('.jpg', '.png')
        if not output_path.endswith('.png'):
            raise ValueError("Output image must be saved as PNG")

        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_image(reveal_tensor, output_path)
    else:
        raise ValueError("Unsupported secret format")
    
    print(f"Decoded secret from {stego_path} into {output_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SteganoModel Data Hiding Tool")
    subparsers = parser.add_subparsers(dest="command")

    encode_parser = subparsers.add_parser("encode", help="Embed secret into cover image")
    encode_parser.add_argument("cover", help="Path to cover image")
    encode_parser.add_argument("--stego", default="dist/stego_image.png", help="Path to save stego image")
    encode_parser.add_argument("secret", help="Path to secret image or text file")

    decode_parser = subparsers.add_parser("decode", help="Extract secret from stego image")
    decode_parser.add_argument("stego", help="Path to stego image")
    decode_parser.add_argument("--output", default="dist/reveal_image.png", help="Path to save extracted secret")

    args = parser.parse_args()

    if args.command == "encode":
        encode(args.cover, args.stego, args.secret)
    elif args.command == "decode":
        decode(args.stego, args.output)
    else:
        parser.print_help()