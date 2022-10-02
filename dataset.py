import pickle
from PIL import Image
import torch
from utils import clean, clean_str
from torch.utils.data import Dataset

from transformers import AutoTokenizer

cat2Encoder = {'자연관광지': 0, '육상 레포츠': 1, '음식점': 2, '축제': 3, '역사관광지': 4, '문화시설': 5, '휴양관광지': 6, '숙박시설': 7, '공연/행사': 8, '쇼핑': 9, '체험관광지': 10, '복합 레포츠': 11, '건축/조형물': 12, '관광자원': 13, '수상 레포츠': 14, '산업관광지': 15, '항공 레포츠': 16, '레포츠소개': 17}
cat2Decoder = {0: '자연관광지', 1: '육상 레포츠', 2: '음식점', 3: '축제', 4: '역사관광지', 5: '문화시설', 6: '휴양관광지', 7: '숙박시설', 8: '공연/행사', 9: '쇼핑', 10: '체험관광지', 11: '복합 레포츠', 12: '건축/조형물', 13: '관광자원', 14: '수상 레포츠', 15: '산업관광지', 16: '항공 레포츠', 17: '레포츠소개'}

def labelEncoder(label) -> list:
    with open('label_encoder.pickle','rb') as f:
        label_dic = pickle.load(f)
    return [label_dic[i] for i in label]

def labelDecoder(cat) -> list:
    with open('label_decoder.pickle','rb') as f:
        label_dic = pickle.load(f)    
    return [label_dic[i] for i in cat]

    
class ImageDataset(Dataset):
    def __init__(self, csv_file, transforms, prediction=False):
        self.img_path = csv_file["img_path"]
        self.transforms = transforms
        self.prediction = prediction
        
        if not self.prediction:
            self.labels = labelEncoder(csv_file["cat3"])
        

    def __getitem__(self, idx):
        img = Image.open("./open"+self.img_path[idx][1:])
        img = img.convert("RGB")
        img = self.transforms(img)
        
        if self.prediction:
            return img, None
        else:
            return img, self.labels[idx]
        
    def __len__(self):
        return len(self.labels)

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, test=False):
        
        pass


class NLDataset(Dataset):
    def __init__(self, csv_file, prediction=False):
        self.overview = [clean_str(clean(i)) for i in csv_file["overview"]]
        self.cat2 = csv_file["cat2"]
        self.cat2Encoder = {v: i for i, v in enumerate(self.cat2.unique())}
        self.cat2Decoder = {i: v for i, v in enumerate(self.cat2.unique())}
        self.prediction = prediction
        if not prediction:
            self.labels = labelEncoder(csv_file["cat3"])
    
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

    def __getitem__(self, idx):
        dic = self.tokenizer.encode_plus(self.overview[idx],
                    add_special_tokens=True,
                    max_length=512,
                    pad_to_max_length='longest',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    ) 
        input_ids = dic['input_ids'].squeeze(0)
        attention_mask = dic['attention_mask'].squeeze(0)
        token_type_ids = dic['token_type_ids'].squeeze(0)
        cat2 = cat2Encoder[self.cat2[idx]]
        label = self.labels[idx] if not self.prediction else None
                            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'cat2': cat2,
            'label': label
        }
    
    def __len__(self):
        return len(self.labels)
    
