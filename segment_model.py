import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from functions import *
from model import build_unet
#from utils import create_dir, seeding 



checkpoint_path = "models/Unet_model.pth"

""" Load the checkpoint """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
    
time_taken = []

seeding(42)

""" Folders """
#create_dir("results")

#Unet_Pytorch=====================================================================
def segment_Unet(path_dir,path_segmented):
    H = 512
    W = 512
    size = (W, H)
    #files=sorted(os.listdir(test_path))
    #num_image=len(files)
    for i, (x, y) in tqdm(enumerate(zip(path_dir,path_dir)), total=len(path_dir)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        src = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        image = cv2.resize(src, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)


        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)


            # score = calculate_metrics(y, pred_y)
            # metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        #ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [pred_y * 255], axis=1
        )
        cv2.imwrite(os.path.join(path_segmented,name+"_.png"), cat_images)

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)