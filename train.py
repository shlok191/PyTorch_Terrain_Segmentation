import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm  # Install tqdm first: pip install tqdm

from dataset import LandCoverDataset
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Defining hyperparameters
batch_size = 8
num_epochs = 10
learning_rate = 1e-2
img_dims = (512, 512)
pin_memory = True
checkpoint_dir = "checkpoints"  # Directory to save checkpoints

# Define scaler for mixed precision training
scaler = GradScaler()


def train(loader, model, optimizer, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, leave=True)
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device=device)
        masks = masks.to(device=device)
        optimizer.zero_grad()
        # Use autocast to perform operations in half precision
        with autocast():
            predictions = model(images)
            loss = loss_fn(predictions, masks)
        # Scale the loss to prevent underflow or overflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        # Update progress bar with loss
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)


def main():
    # Load dataset
    dataset = LandCoverDataset()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    # Define model, optimizer, and loss function
    model = UNet(in_channels=3, out_channels=64, class_count=3, layers=4).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train(loader, model, optimizer, loss_fn, epoch)


if __name__ == "__main__":
    main()
