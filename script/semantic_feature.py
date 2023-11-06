import os
import math
import torch
import clip
import numpy as np
from PIL import Image

import time
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    directory_vevo = "../dataset/vevo/"
    
    for fname in sorted(os.listdir(directory_vevo)):
        if fname.endswith(".mp4"):
            print("id: ", fname[:-4])
            directory_vevo_jpg = os.path.join( "../dataset/vevo_frame/", fname[:-4] )
            file_names = os.listdir(directory_vevo_jpg)
            sorted_file_names = sorted(file_names)
            output_path = "../dataset/vevo_semantic/" + fname[:-4] + ".npy"
            features = torch.cuda.FloatTensor(len(sorted_file_names), 768).fill_(0)
            for idx, file_name in enumerate(sorted_file_names):
                fpath = os.path.join( directory_vevo_jpg, file_name )
                image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)          
                with torch.no_grad():
                    image_features = model.encode_image(image)
                features[idx] = image_features[0]
            features = features.cpu().numpy()
            np.save(output_path, features)
if __name__ == "__main__":
    main()
