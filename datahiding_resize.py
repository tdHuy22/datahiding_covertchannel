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
MAX_LENGTH = 1080

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

def scale_image(img, target_size):
    img_width, img_height = img.size
    if img_width > target_size or img_height > target_size:
        scale = min(target_size / img_width, target_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        print(f"Resizing image from {img.size} to {new_size}")
        img = img.resize(new_size, 1)
    return img

def get_image_ratio(img):
    width, height = img.size
    if width > height:
        ratio = width / height
    else:
        ratio = height / width
    return ratio

def save_image(tensor, save_path):
    tensor_image = torch.clamp(tensor, 0, 1)
    image = to_pil_image(tensor_image.squeeze(0).cpu()).convert("RGB")
    image.save(save_path)

# Encode function
def encode(cover_path, stego_path, secret_path, is_resize):
    if secret_path.endswith('.jpg') or secret_path.endswith('.png'):
        cover_img = Image.open(cover_path).convert("RGB")
        secret_img = Image.open(secret_path).convert("RGB")
        cover_size = cover_img.size
        secret_size = secret_img.size

        print(f"Cover image size (width x height): {cover_size}")
        print(f"Secret image size (width x height): {secret_size}")

        if cover_size[0] > MAX_LENGTH or cover_size[1] > MAX_LENGTH:
            print("WARNING: Cover image is too large (> 1080 pixels) that it may take a very long time to encode.")
            decision = input("Press [Y/y] to resize or [N/n] to continue: ")
            if decision.lower() == 'y':
                cover_img = scale_image(cover_img, MAX_LENGTH)
        cover_size = cover_img.size
        if is_resize:
            secret_img = secret_img.resize(cover_size, 1)
        else:
            if cover_size[0] < secret_size[0] or cover_size[1] < secret_size[1]:
                print("Resizing secret image to fit cover image")
                if cover_ratio > secret_ratio:
                    secret_img = scale_image(secret_img, int(MAX_LENGTH / cover_ratio))
                else:
                    secret_img = scale_image(secret_img, MAX_LENGTH)
        
    elif secret_path.endswith('.txt'):
        cover_img = Image.open(cover_path).convert("RGB")
        cover_size = cover_img.size

        print(f"Cover image size (width x height): {cover_size}")
        if cover_size[0] > MAX_LENGTH or cover_size[1] > MAX_LENGTH:
            print("WARNING: Cover image is too large (> 1080 pixels) that it may take a very long time to encode.")
            decision = input("Press [Y/y] to resize or [N/n] to continue: ")
            if decision.lower() == 'y':
                cover_img = scale_image(cover_img, MAX_LENGTH)
                cover_size = cover_img.size
                print(f"Resized cover image size (width x height): {cover_size}")

        lines = tti.text_lines(secret_path)
        secret_img = tti.lines_image(lines, image_size=cover_size)
    else:
        raise ValueError("Unsupported secret format")
    
    cover_tensor = transform(cover_img).unsqueeze(0).to(DEVICE)
    secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)

    _, _, secret_h, secret_w = secret_tensor.shape
    _, _, cover_h, cover_w = cover_tensor.shape

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
    encode_parser.add_argument("--resize", action="store_true", help="Resize secret image to cover image size")

    decode_parser = subparsers.add_parser("decode", help="Extract secret from stego image")
    decode_parser.add_argument("stego", help="Path to stego image")
    decode_parser.add_argument("--output", default="dist/reveal_image.png", help="Path to save extracted secret")

    args = parser.parse_args()

    if args.command == "encode":
        encode(args.cover, args.stego, args.secret, args.resize)
    elif args.command == "decode":
        decode(args.stego, args.output)
    else:
        parser.print_help()