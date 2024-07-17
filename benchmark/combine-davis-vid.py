import cv2
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--TTA", type=bool, default=False)
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_254_psnr_36.3847.pt") # network_base
parser.add_argument("--path", type=str, default="/home/kim/ssd/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/JPEGImages/480p/")
parser.add_argument("--id", type=str, default="breakdance-flare")

time_interval = 2
H, W = 480, 832*2
FPS = 10.0
# FPS = 5.0
INTERPOLATE4X = True
# INTERPOLATE4X = False


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outVid = cv2.VideoWriter('./video/combined_output.mp4', fourcc, FPS, (W,H))

cap1 = cv2.VideoCapture("./video/output.mp4")
cap2 = cv2.VideoCapture("/home/kim/Desktop/ssd/EMA-VFI/video/output.mp4")

# while cap1.isOpened():
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        frame = cv2.hconcat([frame1, frame2])
        outVid.write(frame)

    else:
        break
    
cap1.release()
cap2.release()
outVid.release()
cv2.destroyAllWindows()     