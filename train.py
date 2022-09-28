import os
import pandas as pd
from PIL import Image
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import set_arguments, control_randomness
from dataset import ImageDataset, MultiModalDataset

from nts_net import model
from nts_net.utils import init_log, progress_bar

# https://dacon.io/competitions/official/235978/codeshare/6565?page=1&dtype=recent

# https://ndb796.tistory.com/373

# https://paperswithcode.com/task/multi-modal-document-classification
# https://github.com/nicolalandro/ntsnet-cub200

SAVE_FREQ = 1
PROPOSAL_NUM = 6
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
    
    
    nets = model.attention_net(topN=PROPOSAL_NUM, num_classes=128, device=device)

    if args.resume:
            ckpt = torch.load(args.ckpt)
            nets.load_state_dict(ckpt['net_state_dict'])
            start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0
        
    nets = nets.to(device)
    creterion = torch.nn.CrossEntropyLoss()
    
    if args.prediction:
        pass
    else:
        trainer(train_loader, val_loader, nets, start_epoch, creterion, args.lr, args.epochs, args.momentum, args.weight_decay, device)
    
def trainer(train_loader, val_loader, nets, start_epoch, creterion, LR, EPOCHS, MMT, WD, device):

    # define optimizers
    raw_parameters = list(nets.pretrained_model.parameters())
    part_parameters = list(nets.proposal_net.parameters())
    concat_parameters = list(nets.concat_net.parameters())
    partcls_parameters = list(nets.partcls_net.parameters())

    raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=MMT, weight_decay=WD)

    concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=MMT, weight_decay=WD)

    part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=MMT, weight_decay=WD)

    partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=MMT, weight_decay=WD)

    schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
    
    for epoch in range(start_epoch, EPOCHS):
        for scheduler in schedulers:
            scheduler.step()
        
        nets.train()
        for i, data in enumerate(train_loader):
            img, label = data[0].to(device), data[1].to(device)
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()
            
            _, _, raw_logits, concat_logits, part_logits, _, top_n_prob = nets(img)
            part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size,
                                                                                                  PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()
            
            progress_bar(i, len(train_loader), 'train')
            print(f'Batch {i}/{len(train_loader)}   Loss: {total_loss.item()}')
    
        nets.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                img, label = data[0].to(device), data[1].to(device)
                batch_size = img.size(0)
                _, _, concat_logits, _, _, _, _ = nets(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(val_loader), 'eval test set')
        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))
        ##########################  save model  ###############################
        net_state_dict = nets.state_dict()
        if epoch % SAVE_FREQ == 0:
            torch.save({
                'epoch': epoch,
                # 'train_loss': train_loss,
                # 'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join("./save_model/", '%03d.ckpt' % epoch))
    print('finishing training')
    
if __name__=="__main__":
    args = set_arguments()
    

    if args.resume:
        if args.checkpoint == "":
            assert False
    
    control_randomness(args.seed)
    
    main(args)
