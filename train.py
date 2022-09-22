
import torch
from utils import set_arguments, control_randomness

def trainer():
    # set gpu
    if args.gpu>=0:
        if torch.cuda.is_available():
            device = torch.device('cuda')  
        else:
            assert False
    else:
        device = torch.device('cpu')
    
    
    pass

if __name__=="__main__":
    args = set_arguments()
    

    if args.resume:
        if args.checkpoint == "":
            assert False
    
    control_randomness(args.seed)
    
    trainer(args)