import os
import pandas as pd
from PIL import Image
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import set_arguments, control_randomness
from dataset import ImageDataset, MultiModalDataset

import torch

# pip install coca-pytorch
# pip install vit-pytorch>=0.35.8

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
from coca_pytorch.coca_pytorch import CoCa

def main(args):
    # set gpu
    if args.gpu>=0:
        if torch.cuda.is_available():
            device = torch.device('cuda')  
        else:
            assert False
    else:
        device = torch.device('cpu')
    
    # load data
    
    transform_train = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        # transforms.CenterCrop((448, 448)),
        transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
        transforms.RandomHorizontalFlip(),  # solo se train
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        # transforms.CenterCrop((448, 448)),
        transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
        # transforms.RandomHorizontalFlip(), # solo se train
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_file = pd.read_csv(f"{args.path}/train_samples.csv")
    val_file = pd.read_csv(f"{args.path}/val_samples.csv")
    if args.multimodal:
        train_dataset = MultiModalDataset(train_file, transform_train, prediction=False)
        val_dataset = MultiModalDataset(val_file, transform_test, prediction=False)
    else:
        train_dataset = ImageDataset(train_file, transform_train, prediction=False)
        val_dataset = ImageDataset(val_file, transform_test, prediction=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_woarkers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_woarkers)
    
    
    
    vit = ViT(image_size = 256, patch_size = 32, num_classes = 128, dim = 512, depth = 6, heads = 16, mlp_dim = 1024)
    vit = Extractor(vit, return_embeddings_only = True, detach = False)

    coca = CoCa(
    dim = 512,                     # model dimension
    img_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
    num_tokens = ---,            # number of text tokens
    unimodal_depth = 6,            # depth of the unimodal transformer
    multimodal_depth = 6,          # depth of the multimodal transformer
    dim_head = 64,                 # dimension per attention head
    heads = 8,                     # number of attention heads
    caption_loss_weight = 1.,      # weight on the autoregressive caption loss
    contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
).to(device)
    if args.prediction:
        pass
    else:
        trainer(train_loader, val_loader, coca, args.lr, args.epochs, args.momentum, args.weight_decay, device)


def trainer(train_loader, val_loader, coca, LR, EPOCHS, MMT, WD, device): 
    for epoch in range(0, EPOCHS):
        
        loss = coca(
            text = text,
            images = images,
            return_loss = True  # set this to True to get the full caption + contrastive loss
        )

        loss.backward()
        


    

if __name__=="__main__":
    args = set_arguments()
    

    if args.resume:
        if args.checkpoint == "":
            assert False
    
    control_randomness(args.seed)
    
    main(args)