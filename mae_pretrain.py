import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision import datasets
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# from torch_utils import training_stats
# from torch_utils import custom_ops


from model import *
from utils import setup_seed
import datautils
import PIL

def ddp_setup(rank, world_size):
    '''
        rank: uniquw identifier for each process
        world_size: total number of processes
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def dataaug(data='imagenet'):
    scale_lower = 0.08
    if data == 'cifar10':
        color_dist_s = 0.5
        im_size = 32
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                im_size,
                scale=(scale_lower, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            # datautils.get_color_distortion(s=color_dist_s),
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

def main(rank, world_size, args):

    ddp_setup(rank, world_size)

    writer = SummaryWriter(os.path.join('logs', args.data, 'mae-pretrain'))

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_transform, test_transform = dataaug(data=args.data)
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(args.val_dir, transform=test_transform)


    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_dataset))

    if args.data == 'cifar10':
        model = MAE_ViT(image_size=32, patch_size=2, encoder_layer=12, encoder_head=3, decoder_layer=4, decoder_head=3, mask_ratio=args.mask_ratio).to(rank)
    elif args.data == 'imagenet':
        model = MAE_ViT(image_size=224, patch_size=16, encoder_layer=12, encoder_head=4, decoder_layer=4, decoder_head=4, mask_ratio=args.mask_ratio).to(rank)
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    os.makedirs('ckpt', exist_ok=True)

    model = DDP(model, device_ids=[rank])

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(rank)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(rank)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        if rank == 0:
            torch.save(model.module, args.model_path)
    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='ckpt/vit-t-mae.pt')
    parser.add_argument('--train_dir', type=str, default='../dataset/imagenet100/train')
    parser.add_argument('--val_dir', type=str, default='../dataset/imagenet100/val')
    parser.add_argument('--data', type=str, default='imagenet')

    args = parser.parse_args()
    setup_seed(args.seed)

    world_size = torch.cuda.device_count()
    print('World Size: {}'.format(world_size))
    mp.spawn(main, args=(world_size, args,), nprocs=world_size)