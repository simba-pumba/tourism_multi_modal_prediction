import argparse
import os
import random
import numpy as np
import torch



def control_randomness(seed:int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    print(f'control_seed: {seed}')    


def set_arguments():
    parser = argparse.ArgumentParser(description="AI Prediction Competetion")

    parser.add_argument('-gpu', 
                        default=0,
                        help='-1: cpu')

    parser.add_argument('-path', 
                    default='./open',
                    help='csv file')

    parser.add_argument('-resume', 
                        action='store_true',
                        help='retraining')

    parser.add_argument('-ckpt',
                        help='model weight file name')


    parser.add_argument('-predction', 
                        action='store_true',
                        help='(test / prediction) ')
    
    parser.add_argument('-multimodal', 
                        action='store_true',
                        help='multimodal')
    
    parser.add_argument('-checkpoint',
                        default="", 
                        help='')
 
    parser.add_argument('-img_size', 
                    default=448)
    
    parser.add_argument('-seed', 
                    default=1234)
       
    parser.add_argument('-epochs', 
                    default=50)
    
    parser.add_argument('-lr', 
                    default=3e-4)

    parser.add_argument('-batch', 
                    default=32)

    parser.add_argument('-num_woarkers', 
                    default=0, 
                    type=int)    
    
    return parser.parse_args()
    
    
