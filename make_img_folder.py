import pandas as pd
import os
import shutil


if __name__=="__main__":
    d = pd.read_csv("./open/train.csv")
    for p, c in zip(d.img_path, d.cat3):
        if not os.path.exists(f"./open/data/{c}"):
            os.makedirs(f"./open/data/{c}")
        shutil.copyfile("./open"+p[1:], f"./open/data/{c}/"+ p.split("/")[-1])
    