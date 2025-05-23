import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform
from tkinter import Tk, font
root = Tk()
font_list = font.families()

MIN_FONT_SIZE = 8
TEXT_COLOR_DELTA = 75
# IMAGE_SIZE = 1024
# TEXT_COLOR = (200, 200, 200)  
# BACKGROUND_COLOR = (125, 125, 125)

is_windows = platform.system() == "Windows"
if is_windows:
    if font_list.count("Times New Roman") > 0:
        FONT_PATH = "times"
        # print("Using Times New Roman font (Windows).")
    elif font_list.count("Arial") > 0:
        FONT_PATH = "arial"
        # print("Using Arial font (Windows).")
    else:
        FONT_PATH = ImageFont.load_default()
        # print("Default font will be used (Windows).")
else:
    if font_list.count("Times New Roman") > 0:
        FONT_PATH = "Times New Roman.ttf"
        # print("Using Times New Roman font.")
    elif font_list.count("Arial") > 0:
        FONT_PATH = "Arial.ttf"
        # print("Using Arial font.")
    elif font_list.count("arial"):
        FONT_PATH = "arial.ttf"
        # print("Using arial font.")
    elif font_list.count("times new roman") > 0:
        FONT_PATH = "times new roman.ttf"
        # print("Using times new roman font.")
    else:
        FONT_PATH = ImageFont.load_default()
        # print("Default font will be used.")

def max_line_length(lines):
    idx_line = 0
    max_length = 0
    for i, line in enumerate(lines):
        if len(line) > max_length:
            max_length = len(line)
            idx_line = i
    return idx_line

def scale_font(lines, draw, image_size):
    font_size = MIN_FONT_SIZE
    idx_max_line_len = max_line_length(lines)
    while True:
        font_temp = ImageFont.truetype(FONT_PATH, font_size)
        if not valid_line_width(lines[idx_max_line_len], draw, font_temp, image_size[0]) or not valid_text_height(lines, draw, font_temp, image_size[1]):
            font_size -= 1
            break
        font_size += 1
    font = ImageFont.truetype(FONT_PATH, font_size)
    return font

def text_lines(text_path):
    with open(text_path, 'r') as file:
        text = file.read()
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            lines.remove(line)
    return lines

def valid_line_width(line, draw, font, image_width):
    bbox = draw.textbbox((0, 0), line, font=font)
    text_width = bbox[2] - bbox[0]
    if text_width > image_width:
        print(f"")
        return False
    return True
    
def valid_text_height(lines, draw, font, image_height):
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = bbox[3] - bbox[1]
        total_height += text_height
    if total_height > image_height:
        return False
    return True

def lines_image(lines, image):
    image_array = np.array(image)
    image_size = image.size
    mean_color = np.mean(image_array, axis=(0, 1)).astype(int)
    background_color = tuple(mean_color)
    text_color = tuple(np.clip(mean_color + TEXT_COLOR_DELTA, 0, 255).astype(int))

    img = Image.new('RGB', image_size, color=background_color)
    draw = ImageDraw.Draw(img)

    image_width, image_height = image.size
    max_text_width = image_width/MIN_FONT_SIZE
    max_text_height = image_height/MIN_FONT_SIZE

    if len(lines) > max_text_height:
        raise ValueError(f"Too many lines: {len(lines)}. Maximum allowed: {max_text_height}")
    if len(lines[max_line_length(lines)]) > max_text_width:
        raise ValueError(f"Line too long: {lines[max_line_length(lines)]}. Maximum allowed: {max_text_width}")

    try:
        print(f"Using font: {FONT_PATH}")
        font = scale_font(lines, draw, image_size)
    except Exception as e:
        print(f"Error scaling font: {e}")
        font = ImageFont.load_default()
        print("Using default font.")

    if not valid_text_height(lines, draw, font, image_height):
        raise ValueError("Text height exceeds image height")

    for i, line in enumerate(lines):
        if not valid_line_width(line, draw, font, image_width):
            raise ValueError(f"Line width exceeds image width: {i + 1}: {line}")

        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (image_width - text_width) / 2
        y = (image_height - text_height) / 2 + i * text_height

        draw.text((x, y), line, fill=text_color, font=font)
    
    return img

# def main():
#     text_path = "test/Secret.txt"
#     image_path = "test/1024x1024_Cover.jpg"
#     image_cover = Image.open(image_path).convert("RGB")
#     lines = text_lines(text_path)
#     img = lines_image(lines, image_cover)
#     img.save("text2image_secret_image.jpg")
#     print("Image saved as 'text2image_secret_image.jpg'")

# if __name__ == '__main__':
#     main()