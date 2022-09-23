import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset

    
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
        self.labels = labelEncoder(csv_file["cat3"])
        self.transforms = transforms
        self.prediction = prediction
        

    
    def __getitem__(self, idx):
        img = Image.open("./open"+self.img_path[idx][1:])
        img = img.convert("RGB")
        img = self.transforms(img)
        
        if self.prediction:
            return img
        else:
            return img, torch.LongTensor(self.labels[idx])
        
    def __len__(self):
        return len(self.labels)

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, test=False):
        
        pass
