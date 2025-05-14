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
MODEL_PATH = os.path.join("Model_v4", "stegano_model_epoch_20.pth")
MAX_SIZE = 300
MIN_SIZE = 100

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

def compute_cover_homogeneity(cover_array, edge_ratio=0.5):
    edge_h, edge_w = int(cover_array.shape[0] * edge_ratio), int(cover_array.shape[1] * edge_ratio)

    top_region = cover_array[:edge_h, :, :]
    bottom_region = cover_array[-edge_h:, :, :]
    left_region = cover_array[:, :edge_w, :]
    right_region = cover_array[:, -edge_w:, :]

    edge_std = {
        'top': np.mean(np.std(top_region, axis=(0, 1))),
        'bottom': np.mean(np.std(bottom_region, axis=(0, 1))),
        'left': np.mean(np.std(left_region, axis=(0, 1))),
        'right': np.mean(np.std(right_region, axis=(0, 1)))
    }

    return edge_std

def padding_secret(secret_img, cover_img):
    cover_w, cover_h = cover_img.size
    secret_w, secret_h = secret_img.size

    cover_array = np.array(cover_img)
    edge_std = compute_cover_homogeneity(cover_array)
    pad_width = cover_w - secret_w
    pad_height = cover_h - secret_h

    total_std_x = edge_std['left'] + edge_std['right']
    total_std_y = edge_std['top'] + edge_std['bottom']

    if total_std_x == 0:
        left_pad = pad_width // 2
    else:
        left_pad = int(pad_width * (edge_std['right'] / total_std_x))
    right_pad = pad_width - left_pad

    if total_std_y == 0:
        top_pad = pad_height // 2
    else:
        top_pad = int(pad_height * (edge_std['bottom'] / total_std_y))
    bottom_pad = pad_height - top_pad
    offset = (left_pad, top_pad)
    
    print(f"Padding: left={left_pad}, right={right_pad}, top={top_pad}, bottom={bottom_pad}")
    print(f"Offset: {offset}")

    mean_color = np.mean(cover_array, axis=(0, 1)).astype(int)
    background_color = tuple(mean_color)

    new_img = Image.new('RGB', (cover_w, cover_h), color=background_color)
    new_img.paste(secret_img, offset)
    return new_img    

def save_image(tensor, save_path):
    tensor_image = torch.clamp(tensor, 0, 1)
    image = to_pil_image(tensor_image.squeeze(0).cpu()).convert("RGB")
    image.save(save_path)

# Encode function
def encode(cover_path, stego_path, secret_path, is_resize):
    cover_img = Image.open(cover_path).convert("RGB")
    if secret_path.endswith('.jpg') or secret_path.endswith('.png'):
        secret_img = Image.open(secret_path).convert("RGB")
        cover_size = cover_img.size
        secret_size = secret_img.size

        print(f"Cover image size (width x height): {cover_size}")
        print(f"Secret image size (width x height): {secret_size}")

        if cover_size[0] < secret_size[0] or cover_size[1] < secret_size[1]:
            raise ValueError("Secret image is larger than cover image")
        
        if secret_size[0] < cover_size[0] or secret_size[1] < cover_size[1]:
            secret_img = padding_secret(secret_img, cover_img)

    elif secret_path.endswith('.txt'):
        lines = tti.text_lines(secret_path)
        secret_img = tti.lines_image(lines, cover_img)
    else:
        raise ValueError("Unsupported secret format")
    
    cover_tensor = transform(cover_img).unsqueeze(0).to(DEVICE)
    secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)
    
    cover_h, cover_w = cover_tensor.shape[2], cover_tensor.shape[3]
    secret_h, secret_w = secret_tensor.shape[2], secret_tensor.shape[3]

    if cover_h % 2 != 0:
        cover_h -= 1
    if cover_w % 2 != 0:
        cover_w -= 1
    if secret_h % 2 != 0:
        secret_h -= 1
    if secret_w % 2 != 0:
        secret_w -= 1
    cover_tensor = F.interpolate(cover_tensor, size=(cover_h, cover_w), mode='bilinear', align_corners=False)
    secret_tensor = F.interpolate(secret_tensor, size=(secret_h, secret_w), mode='bilinear', align_corners=False)

    if cover_h < MIN_SIZE or cover_w < MIN_SIZE:
        raise ValueError("Cover image is too small")

    block_width = None
    block_height = None
    if cover_h > MAX_SIZE or cover_w > MAX_SIZE:
        width_counter = 2
        height_counter = 2
        while cover_h // width_counter > MAX_SIZE:
            width_counter += 2
        while cover_w // height_counter > MAX_SIZE:
            height_counter += 2

        block_width = cover_h // width_counter
        block_height = cover_w // height_counter

    with torch.no_grad():
        if block_width is not None and block_height is not None:
            stego_tensor = torch.zeros_like(cover_tensor)
            for i in range(0, cover_h, block_height):
                for j in range(0, cover_w, block_width):
                    cover_block = cover_tensor[:, :, i:i + block_height, j:j + block_width]
                    secret_block = secret_tensor[:, :, i:i + block_height, j:j + block_width]
                    stego_block, _ = model(cover_block, secret_block)
                    stego_tensor[:, :, i:i + block_height, j:j + block_width] = stego_block
        else:
            stego_tensor, _ = model(cover_tensor, secret_tensor)
        stego_tensor = F.interpolate(stego_tensor, size=(cover_h, cover_w), mode='bilinear', align_corners=False)
    
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
    
    stego_h, stego_w = stego_tensor.shape[2], stego_tensor.shape[3]

    block_width = None
    block_height = None
    if stego_h > MAX_SIZE or stego_w > MAX_SIZE:
        width_counter = 2
        height_counter = 2
        while stego_h // width_counter > MAX_SIZE:
            width_counter += 2
        while stego_w // height_counter > MAX_SIZE:
            height_counter += 2

        block_width = stego_h // width_counter
        block_height = stego_w // height_counter

    # Forward pass with secret_extractor
    with torch.no_grad():
        if block_width is not None and block_height is not None:
            reveal_tensor = torch.zeros_like(stego_tensor)
            for i in range(0, stego_h, block_height):
                for j in range(0, stego_w, block_width):
                    stego_block = stego_tensor[:, :, i:i + block_height, j:j + block_width]
                    reveal_block = model.secret_extractor(stego_block)
                    reveal_tensor[:, :, i:i + block_height, j:j + block_width] = reveal_block
        else:
            reveal_tensor = model.secret_extractor(stego_tensor)
        reveal_tensor = F.interpolate(reveal_tensor, size=(stego_h, stego_w), mode='bilinear', align_corners=False)
    
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