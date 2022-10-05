import os
import pandas as pd
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils import set_arguments, control_randomness
from dataset import ImageDataset, NLDataset, MultiModalDataset
from models import KobertUsingCat2, Kobert
from transformers import AutoConfig
# FOR VISION
from PIL import Image
from torchvision import transforms

# OPTIMIZER FROM TRANSFORMERS
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

warmup_ratio = 0.1
max_grad_norm = 1
log_interval = 50

def main(args):
    # SET GPU
    if args.gpu>=0:
        if torch.cuda.is_available():
            device = torch.device('cuda')  
        else:
            assert False
    else:
        device = torch.device('cpu')
        
    train_dataset = NLDataset(args.train_path, device, prediction=False)
    val_dataset = NLDataset(args.val_path, device, prediction=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, num_workers=args.num_workers)
    # model = KobertUsingCat2(AutoConfig.from_pretrained("monologg/kobert")).to(device)
    model = Kobert(AutoConfig.from_pretrained("monologg/kobert")).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    t_total = len(train_dataloader) * args.epochs
    

    t_total = len(train_dataloader) * args.epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


    # train_history=[]
    val_history=[]
    # loss_history=[]
    max_val_acc = 0.0
    
    dir_name = os.path.join(args.save_path, f"Kobert_{random.randrange(10000,99000)}")
    os.mkdir(dir_name)
    
    for epoch in range(args.epochs):
        train_acc = 0.0
        val_acc = 0.0
        model.train()
        for batch_id, inputs in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss, logits = model(**inputs)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            
            train_acc += calc_accuracy(logits, inputs['labels'])
            
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(epoch+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                # train_history.append(train_acc / (batch_id+1))
                # loss_history.append(loss.data.cpu().numpy())
                
        print("epoch {} train acc {}".format(epoch+1, train_acc / (batch_id+1)))
        #train_history.append(train_acc / (batch_id+1))
        
        model.eval()
        for batch_id, inputs in enumerate(tqdm(val_dataloader)):
            loss, logits = model(**inputs)
            val_acc += calc_accuracy(logits, inputs['labels'])
        val_acc = val_acc / (batch_id+1)
        print("epoch {} test acc {}".format(epoch+1, val_acc))
        if val_acc> max_val_acc:
            max_val_acc = val_acc
            
            torch.save(model.state_dict(), 'model.pth')
            model.load_state_dict(torch.load('model.pth'))

            torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(dir_name, f"Kobert_epoch_{epoch+1}_val_acc_{max_val_acc}.pth"))
            
        val_history.append(val_acc)
    

def test(args):
    pass

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

if __name__ == "__main__":
    args = set_arguments()
    

    if args.resume:
        if args.checkpoint == "":
            assert False
    
    # control_randomness(args.seed)
    
    if args.prediction:
        test(args)
    else:
        main(args)
        
