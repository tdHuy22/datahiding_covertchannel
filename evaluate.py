import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from torch.nn.parallel import DataParallel 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 8
IMAGE_SIZE = 256
AR = 0.65
SIGMA = 0.03
WORKER = 4 
LOG_FILE = 'log.txt'
ANGLE = [0, 90, 180, 270]
MODEL_V2_PATH = "/kaggle/input/model_v2/pytorch/default/1/stegano_model_epoch_20.pth"
MODEL_V4_PATH = "Model_v4/stegano_model_epoch_20.pth"
MODEL_V5_PATH = "Model_v5/stegano_model_epoch_20.pth"

class DilatedInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedInceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, dilation=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=2, dilation=2),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=3, dilation=3),
        )
        self.conv_cat = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.conv_cat(out)
        return out

class SecretEmbedder(nn.Module):
    def __init__(self, concat_channels = 6):
        super(SecretEmbedder, self).__init__()
        self.concat_channels = concat_channels
    
    def forward(self, x1, x2):
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)
        if x.size(1) != self.concat_channels:
            raise ValueError(f"Expected {self.concat_channels} channels, but got {x.size(1)}")
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels = 6, base_channels = 64):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated2 = DilatedInceptionModule(base_channels, base_channels * 2)        # 64 -> 128
        self.dilated3 = DilatedInceptionModule(base_channels * 2, base_channels * 4)    # 128 -> 256
        self.dilated4 = DilatedInceptionModule(base_channels * 4, base_channels * 8)    # 256 -> 512
        self.dilated5 = DilatedInceptionModule(base_channels * 8, base_channels * 8)    # 512 -> 512
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.dilated2(enc1)
        enc3 = self.dilated3(enc2)
        enc4 = self.dilated4(enc3)
        enc5 = self.dilated5(enc4)
        return enc1, enc2, enc3, enc4, enc5

class StegoReconstructor(nn.Module):
    def __init__(self, base_channels = 64, out_channels = 3):
        super(StegoReconstructor, self).__init__()
        self.dilated6 = DilatedInceptionModule(base_channels * 8, base_channels * 8)   # 512 -> 512
        self.dilated7 = DilatedInceptionModule(base_channels * 16, base_channels * 4)  # 1024 -> 256
        self.dilated8 = DilatedInceptionModule(base_channels * 8, base_channels * 2)   # 512 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Đầu ra: N × N × 3 (stego)
        )
    
    def forward(self, enc1, enc2, enc3, enc4, enc5):
        dec5 = self.dilated6(enc5)
        dec4 = self.dilated7(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dilated8(torch.cat([dec4, enc3], dim=1))
        dec2 = self.conv2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.conv_out(torch.cat([dec2, enc1], dim=1))
        return dec1

class SecretExtractor(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, out_channels = 3):
        super(SecretExtractor, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Đầu ra: N × N × 3 (secret)
        )
    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_out(x)
        return x

class SteganoModel(nn.Module):
    def __init__(self, in_channels = 6, base_channels = 64, out_channels = 3):
        super(SteganoModel, self).__init__()
        self.secret_embedder = SecretEmbedder(concat_channels=in_channels)
        self.feature_extractor = FeatureExtractor(in_channels=in_channels, base_channels=base_channels)
        self.stego_reconstructor = StegoReconstructor(base_channels=base_channels, out_channels=out_channels)
        self.secret_extractor = SecretExtractor(in_channels=in_channels // 2, base_channels=base_channels, out_channels=out_channels)


    def forward(self, cover_image, secret_image):
        # Embed secret into cover image
        embedded = self.secret_embedder(cover_image, secret_image)
        
        # Extract features from embedded image
        enc1, enc2, enc3, enc4, enc5 = self.feature_extractor(embedded)
        
        # Reconstruct stego image
        stego_image = self.stego_reconstructor(enc1, enc2, enc3, enc4, enc5)
        
        # Extract secret from stego image
        extracted_secret = self.secret_extractor(stego_image)
        
        return stego_image, extracted_secret

def load_model_v2():
    model = SteganoModel()
    model.load_state_dict(torch.load(MODEL_V2_PATH, map_location=DEVICE, weights_only=True))
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(DEVICE)
    return model

def load_model_v4():
    model = SteganoModel()
    model.load_state_dict(torch.load(MODEL_V4_PATH, map_location=DEVICE, weights_only=True))
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(DEVICE)
    return model

def load_model_v5():
    model = SteganoModel()
    model.load_state_dict(torch.load(MODEL_V5_PATH, map_location=DEVICE, weights_only=True))
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(DEVICE)
    return model

class NoiseTransform(nn.Module):
    def __init__(self, sigma=SIGMA):
        super(NoiseTransform, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x

class AttackTransform(nn.Module):
    def __init__(self, angles=ANGLE):
        super(AttackTransform, self).__init__()
        self.angles = angles

    def forward(self, stego, secret):
        idx = random.randint(0, len(self.angles) - 1)
        angle = self.angles[idx]
        flip_h = (idx % 2 == 0)
        flip_v = (idx % 3 == 0)

        if flip_h:
            stego = v2.functional.hflip(stego)
            secret = v2.functional.hflip(secret)
        if flip_v:
            stego = v2.functional.vflip(stego)
            secret = v2.functional.vflip(secret)
        
        stego = v2.functional.rotate(stego, angle, expand=False)
        secret = v2.functional.rotate(secret, angle, expand=False)

        return stego, secret

noise_transform = NoiseTransform(sigma=SIGMA)
attack_transform = AttackTransform(angles=ANGLE)

class SteganographyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

        self.image_filenames = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPEG')]
        self.pairs = self.create_pairs(self.image_filenames)

        print(f"Created {len(self.pairs)} image pairs from {len(self.image_filenames)} images.")
        if len(self.image_filenames) % 2 != 0:
            print("Warning: The number of images is odd, one image will be ignored.")

    def create_pairs(self, image_filenames):
        pairs = []
        random.shuffle(image_filenames)
        for i in range(0, len(image_filenames), 2):
            if i + 1 < len(image_filenames):
                pairs.append((image_filenames[i], image_filenames[i + 1]))
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        cover_image_path, secret_image_path = self.pairs[idx]
        cover_image = Image.open(cover_image_path).convert('RGB')
        secret_image = Image.open(secret_image_path).convert('RGB')

        if self.transform:
            cover_image = self.transform(cover_image)
            secret_image = self.transform(secret_image)
        return cover_image, secret_image

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
])

def compute_psnr(cover_tensor, stego_tensor, max_value=1.0):
    mse = F.mse_loss(cover_tensor, stego_tensor, reduction='mean')
    if mse == 0:
        mse = 100
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()

def compute_ssim(cover_tensor, stego_tensor, max_value=1.0, k1=0.01, k2=0.03):
    ssim_vals = []
    for i in range(cover_tensor.size(0)):
        cover = cover_tensor[i]
        stego = stego_tensor[i]
        L = max_value
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        c3 = c2 / 2

        mean_cover = torch.mean(cover)
        mean_stego = torch.mean(stego)

        var_cover = torch.var(cover)
        var_stego = torch.var(stego)
        var_cover_sq = torch.sqrt(var_cover)
        var_stego_sq = torch.sqrt(var_stego)

        cov = torch.mean((cover - mean_cover) * (stego - mean_stego))

        l = (2 * mean_cover * mean_stego + c1) / (mean_cover ** 2 + mean_stego ** 2 + c1)
        c = (2 * var_cover_sq * var_stego_sq + c2) / (var_cover + var_stego + c2)
        s = (cov + c3) / (var_cover_sq * var_stego_sq + c3)
        ssim_vals.append(l * c * s)
    return (sum(ssim_vals) / len(ssim_vals)).item()

def plot_batch_metrics(dataset_name, psnr_cover, ssim_cover, psnr_secret, ssim_secret):
    batches = np.arange(1, len(psnr_cover) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot PSNR
    ax1.plot(batches, psnr_cover, label='PSNR Cover', color='skyblue', marker='o')
    ax1.plot(batches, psnr_secret, label='PSNR Secret', color='lightcoral', marker='s')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title(f'PSNR per Batch - {dataset_name}')
    ax1.legend()
    ax1.grid(True)

    # Plot SSIM
    ax2.plot(batches, ssim_cover, label='SSIM Cover', color='lightgreen', marker='o')
    ax2.plot(batches, ssim_secret, label='SSIM Secret', color='gold', marker='s')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM per Batch - {dataset_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_batch_metrics.png')
    plt.close()

def testing(model, test_loader, device, dataset_name, attack=True):
    model.eval()
    psnr_cover_list = []
    ssim_cover_list = []
    psnr_secret_list = []
    ssim_secret_list = []

    progress_bar = tqdm(test_loader, desc=f"Testing {dataset_name}", unit="batch")
    with torch.no_grad():
        for i, (cover, secret) in enumerate(progress_bar):
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR and attack
            
            stego_clean, _ = model(cover_clean, secret_clean)
            if is_attack:
                attacked_stego, attacked_secret = attack_transform(stego_clean, secret_clean)
                attacked_stego = noise_transform(attacked_stego).to(device)
            else:
                attacked_stego = stego_clean.to(device)
                attacked_secret = secret_clean.to(device)
            model_module = model.module if isinstance(model, DataParallel) else model
            revealed = model_module.secret_extractor(attacked_stego)

            psnr_cover = compute_psnr(cover_clean, stego_clean)
            ssim_cover = compute_ssim(cover_clean, stego_clean)
            psnr_secret = compute_psnr(attacked_secret, revealed)
            ssim_secret = compute_ssim(attacked_secret, revealed)

            psnr_cover_list.append(psnr_cover)
            ssim_cover_list.append(ssim_cover)
            psnr_secret_list.append(psnr_secret)
            ssim_secret_list.append(ssim_secret)

    # Compute averages for logging
    avg_psnr_cover = sum(psnr_cover_list) / len(psnr_cover_list)
    avg_ssim_cover = sum(ssim_cover_list) / len(ssim_cover_list)
    avg_psnr_secret = sum(psnr_secret_list) / len(psnr_secret_list)
    avg_ssim_secret = sum(ssim_secret_list) / len(ssim_secret_list)

    print(f"{dataset_name} - Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}")
    print(f"{dataset_name} - Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}")

    with open(LOG_FILE, 'a') as f:
        f.write(f"{dataset_name} - Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}\n")
        f.write(f"{dataset_name} - Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}\n")
        f.flush()

    return dataset_name, psnr_cover_list, ssim_cover_list, psnr_secret_list, ssim_secret_list

if __name__ == "__main__":
    coco_dataset_path = "/kaggle/input/2017-2017/test2017/test2017"
    pascal_voc_dataset_path = "/kaggle/input/pascal-voc-2012-dataset/VOC2012_test/VOC2012_test/JPEGImages"
    image_net_dataset_path = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
    
    cover_test_dataset = SteganographyDataset(coco_dataset_path, transform=transform)
    pascal_voc_test_dataset = SteganographyDataset(pascal_voc_dataset_path, transform=transform)
    image_net_test_dataset = SteganographyDataset(image_net_dataset_path, transform=transform)

    coco_test_loader = DataLoader(cover_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER)
    pascal_voc_test_loader = DataLoader(pascal_voc_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER)
    image_net_test_loader = DataLoader(image_net_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER)

    model = load_model_v5()
    plot_batch_metrics(*testing(model, pascal_voc_test_loader, DEVICE, "Pascal VOC", attack=False))
    plot_batch_metrics(*testing(model, image_net_test_loader, DEVICE, "ImageNet", attack=False))
    plot_batch_metrics(*testing(model, coco_test_loader, DEVICE, "COCO", attack=False))