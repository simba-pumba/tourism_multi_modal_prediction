import argparse
import os
import random
import numpy as np
import torch
import re



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


    parser.add_argument('-prediction', 
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
                    default=0.001)

    parser.add_argument('-batch', 
                    default=32,
                    type=int)

    parser.add_argument('-num_woarkers', 
                    default=0, 
                    type=int)    
 
    parser.add_argument('-momentum', 
                    default=0.9, 
                    type=float)   

    parser.add_argument('-weight_decay', 
                    default=1e-4, 
                    type=float)      
    return parser.parse_args()
    
    
# 출처 https://jsikim1.tistory.com/213
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } 


def clean(text):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()


def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[^가-힣]', ' ', text) # 이모티콘, 특수 기호 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
    text = re.sub('\n', '.', string=text)
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip()
    text = text.lstrip()
    return text 
