import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import random

ANGLES = [90, 180, 270]
SIGMA = 0.03

rotation_transform = v2.Lambda(lambda image: v2.functional.rotate(image, angle=random.choice(ANGLES), expand=False))

attack_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    rotation_transform,
    v2.GaussianNoise(sigma=SIGMA),
    v2.ToPILImage(),
])

# def attack_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = image.transpose(Image.FLIP_LEFT_RIGHT)
#     iamge = image.transpose(Image.FLIP_TOP_BOTTOM)
#     image = image.transpose(Image.ROTATE_90)
#     return image

def attack_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # Apply the attack transform
    attacked_image = attack_transform(image)
    return attacked_image

if __name__ == "__main__":
    image_path = "/Users/macos/Documents/MasterUIT/IT2033.CH190_DataHiding/covertchannel/dist/stego_image_v5.png"  # Update with your path
    flipped_image = attack_image(image_path)
    flipped_image.save("/Users/macos/Documents/MasterUIT/IT2033.CH190_DataHiding/covertchannel/dist/attacked_stego_image_v5.png")  # Save the flipped image
