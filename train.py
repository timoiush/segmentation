import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from datasets.datasets import DriveDataset
from models.networks import UNet


file_path = 'datasets/drive_data'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 4
dataset = DriveDataset(filepath=file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = UNet().to(device)

n_epochs = 50
lr = 1e-3
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.2)


def main():
    train_loss = []
    model.train()
    for epoch in range(1, n_epochs+1):
        print(f'Epoch {epoch}/{n_epochs}')
        print('-'*10)
        epoch_loss = 0.0

        for items in dataloader:
            imgs, masks = items['image'].to(device), items['mask'].to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(dataloader)
        train_loss.append(epoch_loss)
        print(f'Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    main()