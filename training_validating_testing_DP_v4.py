import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF2

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
LR = 0.0002
AR = 0.4
SIGMA = 0.02
WORKER = 4 
LAMBDA = 1
OLD_LOG_FILE = 'old_log.txt'
LOG_FILE = 'log.txt'
ANGLE = [0, 90, 180, 270]
CONTINUOUS = False
CHECKPOINT = ''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CoverImage Dataset
train_dataset_path = '/kaggle/input/2017-2017/train2017/train2017'
valid_dataset_path = '/kaggle/input/2017-2017/val2017/val2017'
test_dataset_path = '/kaggle/input/2017-2017/test2017/test2017'

class SteganographyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

        self.image_filenames = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

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
            stego = TF2.hflip(stego)
            secret = TF2.hflip(secret)
        if flip_v:
            stego = TF2.vflip(stego)
            secret = TF2.vflip(secret)
        
        stego = TF2.rotate(stego, angle, expand=False)
        secret = TF2.rotate(secret, angle, expand=False)

        return stego, secret

noise_transform = NoiseTransform(sigma=SIGMA)
attack_transform = AttackTransform(angles=ANGLE)
        
def loss_end_to_end(cover_images, stego_images, secret_images, reveal_images, lambda_l2=1):
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    l1_loss = l1(cover_images, stego_images)
    l2_loss = l2(secret_images, reveal_images)
    variance_l1_loss = torch.var(torch.abs(cover_images - stego_images)) / cover_images.size(0)
    total_loss = 1/2 * (l1_loss + variance_l1_loss) + lambda_l2*l2_loss
    return total_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, valid_loss, valid_counter, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss": loss,
        "valid_loss": valid_loss,
        "valid_counter": valid_counter,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename=CHECKPOINT, device="cuda"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} does not exist.")

    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float('inf'))
    valid_loss = checkpoint.get("valid_loss", float('inf'))
    valid_counter = checkpoint.get("valid_counter", 0)
    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return start_epoch, loss, valid_loss, valid_counter

def validation(model, val_loader, epoch, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")
    with torch.no_grad():
        for cover, secret in progress_bar:
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR

            stego_clean, _ = model(cover_clean, secret_clean)
            if is_attack:
                attacked_stego, attacked_secret = attack_transform(stego_clean, secret_clean)
                attacked_stego = noise_transform(attacked_stego).to(device)
            else:
                attacked_stego = stego_clean.to(device)
                attacked_secret = secret_clean.to(device)

            model_module = model.module if isinstance(model, DataParallel) else model
            revealed = model_module.secret_extractor(attacked_stego)

            loss = loss_end_to_end(cover_clean, stego_clean, attacked_secret, revealed, lambda_l2=LAMBDA)
            total_loss += loss.item()
            progress_bar.set_postfix(valid_loss=loss.item())
    return total_loss / len(val_loader)

def copy_log_file():
    if os.path.exists(OLD_LOG_FILE):
        with open(OLD_LOG_FILE, 'r') as old_log:
            lines = old_log.readlines()
        with open(LOG_FILE, 'w') as new_log:
            new_log.writelines(lines)
        print(f"Copied log file from {OLD_LOG_FILE} to {LOG_FILE}")
    else:
        print(f"Log file {OLD_LOG_FILE} does not exist. No copy made.")

def training(model, train_loader, val_loader, optimizer, scheduler, scaler, device):
    best_val_loss = float('inf')
    val_patience = 5
    patience_counter = 0

    if CONTINUOUS:
        start_epoch, loss, valid_loss, valid_counter = load_checkpoint(model, optimizer, scheduler, filename=CHECKPOINT, device=device)
        print(f"Resuming training from epoch {start_epoch} with loss {loss} and valid_loss {valid_loss}")
        best_val_loss = valid_loss
        patience_counter = valid_counter
        copy_log_file()
        with open(LOG_FILE, 'a') as f:
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for i, (cover, secret) in enumerate(progress_bar):
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                stego_clean, _ = model(cover_clean, secret_clean)
                if is_attack:
                    attacked_stego, attacked_secret = attack_transform(stego_clean, secret_clean)
                    attacked_stego = noise_transform(attacked_stego).to(device)
                else:
                    attacked_stego = stego_clean.to(device)
                    attacked_secret = secret_clean.to(device)

                model_module = model.module if isinstance(model, DataParallel) else model
                revealed = model_module.secret_extractor(attacked_stego)
                loss = loss_end_to_end(cover_clean, stego_clean, attacked_secret, revealed, lambda_l2=LAMBDA)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step(total_loss_epoch / len(train_loader))

        val_loss = validation(model, val_loader, epoch + 1, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(train_loader):.4f}, Validation_Loss: {val_loss:.4f}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(train_loader):.4f}, Validation_Loss: {val_loss:.4f}\n")
            f.flush()

        model_to_log = model.module if isinstance(model, DataParallel) else model
        state_dict = model_to_log.state_dict()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch + 1}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch + 1}\n")
                f.flush()
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{val_patience}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Validation loss did not improve. Patience counter: {patience_counter}/{val_patience}\n")
                f.flush()

        with open(LOG_FILE, 'a') as f:
            f.write(f"\n")
            f.flush()

        # Save model checkpoint
        torch.save(state_dict, f"stegano_model_epoch_{epoch + 1}.pth")
        save_checkpoint(model_to_log, optimizer, scheduler, epoch + 1, total_loss_epoch / len(train_loader), val_loss, patience_counter, filename=f"checkpoint_epoch_{epoch + 1}.pth")

        if patience_counter >= val_patience:
            print("Early stopping triggered.")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1}\n")
                f.flush()
            break

    # Log the final model
    model_to_log = model.module if isinstance(model, DataParallel) else model
    state_dict = model_to_log.state_dict()

    torch.save(state_dict, "stegano_model_final.pth")
    save_checkpoint(model_to_log, optimizer, scheduler, EPOCHS, total_loss_epoch / len(train_loader), val_loss, patience_counter, filename="checkpoint_final.pth")

    print("Training completed!")

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
    return sum(ssim_vals) / len(ssim_vals)

def testing(model, test_loader, device):
    model.eval()
    total_psnr_cover = 0
    total_ssim_cover = 0
    total_psnr_secret = 0
    total_ssim_secret = 0
    with torch.no_grad():
        for cover, secret in test_loader:
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR
            
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
            total_psnr_cover += psnr_cover
            total_ssim_cover += ssim_cover
            total_psnr_secret += psnr_secret
            total_ssim_secret += ssim_secret
    avg_psnr_cover = total_psnr_cover / len(test_loader)
    avg_ssim_cover = total_ssim_cover / len(test_loader)
    avg_psnr_secret = total_psnr_secret / len(test_loader)
    avg_ssim_secret = total_ssim_secret / len(test_loader)
    print(f"Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}")
    print(f"Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}")
    with open(LOG_FILE, 'a') as f:
        f.write(f"Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}\n")
        f.write(f"Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}\n")
        f.flush()
    return avg_psnr_cover, avg_ssim_cover, avg_psnr_secret, avg_ssim_secret

if __name__ == '__main__':
    train_dataset = SteganographyDataset(train_dataset_path, transform=transform)
    valid_dataset = SteganographyDataset(valid_dataset_path, transform=transform)
    test_dataset = SteganographyDataset(test_dataset_path, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKER,
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER,
        pin_memory=True
    )

    # Khởi tạo model và wrap bằng DataParallel
    model = SteganoModel().to(DEVICE)

    # Khởi tạo optimizer và scaler
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')

    training(model, train_loader, val_loader, optimizer, scheduler, scaler, DEVICE)
    testing(model, test_loader, DEVICE)import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF2

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
LR = 0.0002
AR = 0.4
SIGMA = 0.02
WORKER = 4 
LAMBDA = 1
OLD_LOG_FILE = 'old_log.txt'
LOG_FILE = 'log.txt'
ANGLE = [0, 90, 180, 270]
CONTINUOUS = False
CHECKPOINT = ''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CoverImage Dataset
train_dataset_path = '/kaggle/input/2017-2017/train2017/train2017'
valid_dataset_path = '/kaggle/input/2017-2017/val2017/val2017'
test_dataset_path = '/kaggle/input/2017-2017/test2017/test2017'

class SteganographyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

        self.image_filenames = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]

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
            stego = TF2.hflip(stego)
            secret = TF2.hflip(secret)
        if flip_v:
            stego = TF2.vflip(stego)
            secret = TF2.vflip(secret)
        
        stego = TF2.rotate(stego, angle, expand=False)
        secret = TF2.rotate(secret, angle, expand=False)

        return stego, secret

noise_transform = NoiseTransform(sigma=SIGMA)
attack_transform = AttackTransform(angles=ANGLE)
        
def loss_end_to_end(cover_images, stego_images, secret_images, reveal_images, lambda_l2=1):
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    l1_loss = l1(cover_images, stego_images)
    l2_loss = l2(secret_images, reveal_images)
    variance_l1_loss = torch.var(torch.abs(cover_images - stego_images)) / cover_images.size(0)
    total_loss = 1/2 * (l1_loss + variance_l1_loss) + lambda_l2*l2_loss
    return total_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="checkpoint.pth"):
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
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} does not exist.")

    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return start_epoch, loss

def validation(model, val_loader, epoch, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")
    with torch.no_grad():
        for cover, secret in progress_bar:
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR

            stego_clean, _ = model(cover_clean, secret_clean)
            if is_attack:
                attacked_stego, attacked_secret = attack_transform(stego_clean, secret_clean)
                attacked_stego = noise_transform(attacked_stego).to(device)
            else:
                attacked_stego = stego_clean.to(device)
                attacked_secret = secret_clean.to(device)

            model_module = model.module if isinstance(model, DataParallel) else model
            revealed = model_module.secret_extractor(attacked_stego)

            loss = loss_end_to_end(cover_clean, stego_clean, attacked_secret, revealed, lambda_l2=LAMBDA)
            total_loss += loss.item()
            progress_bar.set_postfix(valid_loss=loss.item())
    return total_loss / len(val_loader)

def copy_log_file():
    if os.path.exists(OLD_LOG_FILE):
        with open(OLD_LOG_FILE, 'r') as old_log:
            lines = old_log.readlines()
        with open(LOG_FILE, 'w') as new_log:
            new_log.writelines(lines)
        print(f"Copied log file from {OLD_LOG_FILE} to {LOG_FILE}")
    else:
        print(f"Log file {OLD_LOG_FILE} does not exist. No copy made.")

def training(model, train_loader, val_loader, optimizer, scheduler, scaler, device):
    best_val_loss = float('inf')
    val_patience = 5
    patience_counter = 0

    if CONTINUOUS:
        start_epoch, loss = load_checkpoint(model, optimizer, scheduler, filename=CHECKPOINT, device=device)
        print(f"Resuming training from epoch {start_epoch} with loss {loss}")
        copy_log_file()
        with open(LOG_FILE, 'a') as f:
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for i, (cover, secret) in enumerate(progress_bar):
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                stego_clean, _ = model(cover_clean, secret_clean)
                if is_attack:
                    attacked_stego, attacked_secret = attack_transform(stego_clean, secret_clean)
                    attacked_stego = noise_transform(attacked_stego).to(device)
                else:
                    attacked_stego = stego_clean.to(device)
                    attacked_secret = secret_clean.to(device)

                model_module = model.module if isinstance(model, DataParallel) else model
                revealed = model_module.secret_extractor(attacked_stego)
                loss = loss_end_to_end(cover_clean, stego_clean, attacked_secret, revealed, lambda_l2=LAMBDA)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step(total_loss_epoch / len(train_loader))

        val_loss = validation(model, val_loader, epoch + 1, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(train_loader):.4f}, Validation_Loss: {val_loss:.4f}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Total_Loss_Epoch: {total_loss_epoch / len(train_loader):.4f}, Validation_Loss: {val_loss:.4f}\n")
            f.flush()

        model_to_log = model.module if isinstance(model, DataParallel) else model
        state_dict = model_to_log.state_dict()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch + 1}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch + 1}\n")
                f.flush()
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{val_patience}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Validation loss did not improve. Patience counter: {patience_counter}/{val_patience}\n")
                f.flush()

        with open(LOG_FILE, 'a') as f:
            f.write(f"\n")
            f.flush()

        # Save model checkpoint
        torch.save(state_dict, f"stegano_model_epoch_{epoch + 1}.pth")
        save_checkpoint(model_to_log, optimizer, scheduler, epoch + 1, total_loss_epoch / len(train_loader), filename=f"checkpoint_epoch_{epoch + 1}.pth")

        if patience_counter >= val_patience:
            print("Early stopping triggered.")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1}\n")
                f.flush()
            break

    # Log the final model
    model_to_log = model.module if isinstance(model, DataParallel) else model
    state_dict = model_to_log.state_dict()

    torch.save(state_dict, "stegano_model_final.pth")
    save_checkpoint(model_to_log, optimizer, scheduler, EPOCHS, total_loss_epoch / len(train_loader), filename="checkpoint_final.pth")

    print("Training completed!")

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
    return sum(ssim_vals) / len(ssim_vals)

def testing(model, test_loader, device):
    model.eval()
    total_psnr_cover = 0
    total_ssim_cover = 0
    total_psnr_secret = 0
    total_ssim_secret = 0
    with torch.no_grad():
        for cover, secret in test_loader:
            cover_clean = cover.to(device)
            secret_clean = secret.to(device)
            is_attack = random.random() < AR
            
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
            total_psnr_cover += psnr_cover
            total_ssim_cover += ssim_cover
            total_psnr_secret += psnr_secret
            total_ssim_secret += ssim_secret
    avg_psnr_cover = total_psnr_cover / len(test_loader)
    avg_ssim_cover = total_ssim_cover / len(test_loader)
    avg_psnr_secret = total_psnr_secret / len(test_loader)
    avg_ssim_secret = total_ssim_secret / len(test_loader)
    print(f"Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}")
    print(f"Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}")
    with open(LOG_FILE, 'a') as f:
        f.write(f"Average PSNR (Cover): {avg_psnr_cover:.4f}, Average SSIM (Cover): {avg_ssim_cover:.4f}\n")
        f.write(f"Average PSNR (Secret): {avg_psnr_secret:.4f}, Average SSIM (Secret): {avg_ssim_secret:.4f}\n")
        f.flush()
    return avg_psnr_cover, avg_ssim_cover, avg_psnr_secret, avg_ssim_secret

if __name__ == '__main__':
    train_dataset = SteganographyDataset(train_dataset_path, transform=transform)
    valid_dataset = SteganographyDataset(valid_dataset_path, transform=transform)
    test_dataset = SteganographyDataset(test_dataset_path, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKER,
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER,
        pin_memory=True
    )

    # Khởi tạo model và wrap bằng DataParallel
    model = SteganoModel().to(DEVICE)

    # Khởi tạo optimizer và scaler
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')

    training(model, train_loader, val_loader, optimizer, scheduler, scaler, DEVICE)
    testing(model, test_loader, DEVICE)