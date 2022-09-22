import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform_train = transforms.Compose([
    # transforms.Resize((600, 600), Image.BILINEAR),
    # transforms.CenterCrop((448, 448)),
    transforms.Resize((448, 448), Image.BILINEAR),
    transforms.RandomHorizontalFlip(),  # solo se train
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def labelEncoder(d):
    with open('label_encoder.pickle','rb') as f:
        label_dic = pickle.load(f)
        
        pass



    
class ImageDataset(Dataset):
    def __init__(self, data_list, test=False):
        pass
    
    def __getitem(self, index):
        pass
        
        # if test:
        # return
        # else:
        # return 
    def __len__(self):
        pass
        # return len(?)

class MultiModalDataset(Dataset):
    def __init__(self, data_list, test=False):
        
        pass