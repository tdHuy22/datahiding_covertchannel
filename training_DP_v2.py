import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel

from PIL import Image
from tqdm import tqdm

import os
import random

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

# Hyperparameters
BATCH_SIZE = 8
IMAGE_SIZE = 256
EPOCHS = 20
LR = 0.0001
WORKER = 4
LAMBDA = 1
LOG_FILE = 'log.txt'
CONTINUOUS = True
CHECKPOINT = '/kaggle/input/covertchannel-dp-v2/checkpoint_epoch_5.pth'

# CoverImage Dataset
dataset_path = '/kaggle/input/2017-2017/train2017/train2017'

class SteganographyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

        self.image_filenames = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

        self.pairs = self.create_pairs(self.image_filenames)

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

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def loss_end_to_end(cover_images, stego_images, secret_images, reveal_images, lambda_l2=1):
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    l1_loss = l1(cover_images, stego_images)
    l2_loss = l2(secret_images, reveal_images)
    variance_l1_loss = torch.var(torch.abs(cover_images - stego_images))
    total_loss = 1/2 * (l1_loss + variance_l1_loss) + lambda_l2*l2_loss
    return total_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename=CHECKPOINT):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename=CHECKPOINT, device="cuda"):
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return start_epoch, loss

def training():
    # Khởi tạo device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo dataset và dataloader
    cover_dataset = SteganographyDataset(dataset_path, transform=transform)
    dataloader = DataLoader(
        cover_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKER,
        pin_memory=True
    )

    # Khởi tạo model và wrap bằng DataParallel
    model = SteganoModel().to(device)

    # Khởi tạo optimizer và scaler
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')

    if CONTINUOUS:
        # Khôi phục từ checkpoint
        start_epoch, loss = load_checkpoint(model, optimizer, scheduler, filename=CHECKPOINT, device=device)
        print(f"Resuming training from epoch {start_epoch} with loss {loss}")
        with open(LOG_FILE, 'w') as f:
            f.write("Continuous training started...\n")
            f.flush()
    else:
        with open(LOG_FILE, 'w') as f:
            f.write("Training started...\n")
            f.flush()
        start_epoch = 0
        print("Starting training from scratch.")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss_epoch = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for i, (cover, secret) in enumerate(progress_bar):
            cover = cover.to(device)
            secret = secret.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                stego, revealed = model(cover, secret)
                loss = loss_end_to_end(cover, stego, secret, revealed, lambda_l2=LAMBDA)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step(total_loss_epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(dataloader):.4f}")
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(dataloader):.4f}\n")
            f.flush()

        model_to_log = model.module if isinstance(model, DataParallel) else model
        state_dict = model_to_log.state_dict()
        torch.save(state_dict, f"stegano_model_epoch_{epoch + 1}.pth")

        save_checkpoint(model_to_log, optimizer, scheduler, epoch + 1, total_loss_epoch / len(dataloader), filename=f"checkpoint_epoch_{epoch + 1}.pth")

    # Log the final model
    model_to_log = model.module if isinstance(model, DataParallel) else model
    state_dict = model_to_log.state_dict()
    torch.save(state_dict, "stegano_model_final.pth")
    save_checkpoint(model_to_log, optimizer, scheduler, EPOCHS, total_loss_epoch / len(dataloader), filename="checkpoint_final.pth")

    print("Training completed!")

if __name__ == '__main__':
    training()