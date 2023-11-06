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
    text = clip.tokenize(["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]).to(device)
    directory_vevo = "../dataset/vevo/"
    directory_vevo_emotion = "../dataset/vevo_emotion/6c_l14p/"

    for fname in sorted(os.listdir(directory_vevo)):
        if fname.endswith(".mp4"):
            print("id: ", fname[:-4])
            emolab_path = os.path.join( directory_vevo_emotion, fname[:-4] + ".lab" )
            directory_vevo_jpg = os.path.join( "../dataset/vevo_frame/", fname[:-4] )
            file_names = os.listdir(directory_vevo_jpg)
            sorted_file_names = sorted(file_names)
            emolist = []
            for file_name in sorted_file_names:
                fpath = os.path.join( directory_vevo_jpg, file_name )
                image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)                
                with torch.no_grad():  
                    logits_per_image, logits_per_text = model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                fp1 = format(probs[0][0], ".4f")
                fp2 = format(probs[0][1], ".4f")
                fp3 = format(probs[0][2], ".4f")
                fp4 = format(probs[0][3], ".4f")
                fp5 = format(probs[0][4], ".4f")
                fp6 = format(probs[0][5], ".4f")
                
                emo_val = str(fp1) +" "+ str(fp2) +" "+ str(fp3) +" "+ str(fp4) +" "+ str(fp5) + " " + str(fp6)
                emolist.append(emo_val)
            
            with open(emolab_path ,'w' ,encoding = 'utf-8') as f:
                f.write("time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob\n")
                for i in range(0, len(emolist) ):
                    f.write(str(i) + " "+emolist[i]+"\n")

if __name__ == "__main__":
    main()
