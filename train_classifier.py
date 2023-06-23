import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm

from model import *
from utils import setup_seed
import datautils
import PIL


def dataaug(data='imagenet'):
    scale_lower = 0.08
    if data == 'cifar':
        color_dist_s = 0.5
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(scale_lower, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            # datautils.get_color_distortion(s=color_dist_s),
            datautils.Clip(),
        ])
        test_transform = train_transform

    elif data == 'imagenet':
        from datautils import GaussianBlur
        color_dist_s = 1.0
        im_size = 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                im_size,
                scale=(scale_lower, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomHorizontalFlip(0.5),
            # datautils.get_color_distortion(s=self.hparams.color_dist_s),
            # GaussianBlur(im_size // 10, 0.5),
            datautils.Clip(),
        ])
        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                im_size,
                scale=(scale_lower, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            # transforms.RandomHorizontalFlip(0.5),
            # datautils.get_color_distortion(s=self.hparams.color_dist_s),
            # GaussianBlur(im_size // 10, 0.5),
            datautils.Clip(),
        ])
        # test_transform = train_transform
    return train_transform, test_transform



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='ckpt/vit-t-classifier-from_scratch.pt')
    parser.add_argument('--train_dir', type=str, default='../dataset/imagenet100/train')
    parser.add_argument('--val_dir', type=str, default='../dataset/imagenet100/val')
    parser.add_argument('--data', type=str, default='imagenet')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    train_transform, test_transform = dataaug(data=args.data)
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(args.val_dir, transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location=device)
        writer = SummaryWriter(os.path.join('logs', args.data, 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', args.data, 'scratch-cls'))
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    os.makedirs('ckpt', exist_ok=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model, args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)