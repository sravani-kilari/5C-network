import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.nested_unet import NestedUNet
from models.attention_unet import AttentionUNet
from dataset import BrainMetastasisDataset  
import argparse

def train_model(model, train_loader, epochs, optimizer, criterion, device):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['nested_unet', 'attention_unet'], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the training on")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.model == 'nested_unet':
        model = NestedUNet()
    else:
        model = AttentionUNet()

    train_dataset = BrainMetastasisDataset('data/preprocessed')  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()  

    train_model(model, train_loader, args.epochs, optimizer, criterion, device)
