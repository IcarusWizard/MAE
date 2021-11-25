import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=8000)
    parser.add_argument('--warmup_epoch', type=int, default=400)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')

    args = parser.parse_args()

    dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4)
    lr = args.base_learning_rate * args.batch_size / 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    for e in range(args.total_epoch):
        losses = []
        for img, label in tqdm(iter(dataloader)):
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / 0.75
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        torch.save(model, args.model_path)