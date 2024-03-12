import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

from pytorch_msssim import ssim_matlab
sys.path.append('../')

# from network19 import Network
from network22 import Network


class YUV_Read():
    def __init__(self, filepath, h, w, format='yuv420', toRGB=True):

        self.h = h
        self.w = w

        self.fp = open(filepath, 'rb')

        if format == 'yuv420':
            self.frame_length = int(1.5 * h * w)
            self.Y_length = h * w
            self.Uv_length = int(0.25 * h * w)
        else:
            pass
        self.toRGB = toRGB

    def read(self, offset_frame=None):
        if not offset_frame == None:
            self.fp.seek(offset_frame * self.frame_length, 0)

        Y = np.fromfile(self.fp, np.uint8, count=self.Y_length)
        U = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
        V = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
        if Y.size < self.Y_length or \
                        U.size < self.Uv_length or \
                        V.size < self.Uv_length:
            return None, False

        Y = np.reshape(Y, [self.w, self.h], order='F')
        Y = np.transpose(Y)

        U = np.reshape(U, [int(self.w / 2), int(self.h / 2)], order='F')
        U = np.transpose(U)

        V = np.reshape(V, [int(self.w / 2), int(self.h / 2)], order='F')
        V = np.transpose(V)

        U = np.array(Image.fromarray(U).resize([self.w, self.h]))
        V = np.array(Image.fromarray(V).resize([self.w, self.h]))

        if self.toRGB:
            Y = Y / 255.0
            U = U / 255.0 - 0.5
            V = V / 255.0 - 0.5

            self.YUV = np.stack((Y, U, V), axis=-1)
            self.RGB = (255.0 * np.clip(yuv2rgb(self.YUV), 0.0, 1.0)).astype('uint8')

            self.YUV = None
            return self.RGB, True
        else:
            self.YUV = np.stack((Y, U, V), axis=-1)
            return self.YUV, True

    def close(self):
        self.fp.close()

class YUV_Write():
    def __init__(self, filepath, fromRGB=True):
        if os.path.exists(filepath):
            print(filepath)
  
        self.fp = open(filepath, 'wb')
        self.fromRGB = fromRGB

    def write(self, Frame):

        self.h = Frame.shape[0]
        self.w = Frame.shape[1]
        c = Frame.shape[2]

        assert c == 3
        if format == 'yuv420':
            self.frame_length = int(1.5 * self.h * self.w)
            self.Y_length = self.h * self.w
            self.Uv_length = int(0.25 * self.h * self.w)
        else:
            pass
        if self.fromRGB:
            Frame = Frame / 255.0
            YUV = rgb2yuv(Frame)
            Y, U, V = np.dsplit(YUV, 3)
            Y = Y[:, :, 0]
            U = U[:, :, 0]
            V = V[:, :, 0]
            U = np.clip(U + 0.5, 0.0, 1.0)
            V = np.clip(V + 0.5, 0.0, 1.0)

            U = U[::2, ::2]  # imresize(U,[int(self.h/2),int(self.w/2)],interp = 'nearest')
            V = V[::2, ::2]  # imresize(V ,[int(self.h/2),int(self.w/2)],interp = 'nearest')
            Y = (255.0 * Y).astype('uint8')
            U = (255.0 * U).astype('uint8')
            V = (255.0 * V).astype('uint8')
        else:
            YUV = Frame
            Y = YUV[:, :, 0]
            U = YUV[::2, ::2, 1]
            V = YUV[::2, ::2, 2]

        Y = Y.flatten()  # the first order is 0-dimension so don't need to transpose before flatten
        U = U.flatten()
        V = V.flatten()

        Y.tofile(self.fp)
        U.tofile(self.fp)
        V.tofile(self.fp)

        return True

    def close(self):
        self.fp.close()

def load_model_checkpoint(model, checkpoint_path):
	param = torch.load(checkpoint_path, map_location='cuda:0')
	layers_to_remove = []
	for key in param:
		if "pretrained_flow_net" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.HW" in key:
			layers_to_remove.append(key)
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/HD_dataset")
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 34.44/0.9737
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 35.28/0.9771
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 32.593
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
load_model_checkpoint(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")   

TTA = args.TTA
name_list = [
    (f'{args.path}/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
    (f'{args.path}/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
    (f'{args.path}/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
    (f'{args.path}/HD1080p_GT/BlueSky.yuv', 1080, 1920),
    (f'{args.path}/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
    (f'{args.path}/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
    (f'{args.path}/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
    (f'{args.path}/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
    (f'{args.path}/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
    (f'{args.path}/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
    (f'{args.path}/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
]
tot = 0.
for data in tqdm(name_list):
    psnr_list = []
    name = data[0]
    h = data[1]
    w = data[2]
    if 'yuv' in name:
        Reader = YUV_Read(name, h, w, toRGB=True)
    else:
        Reader = cv2.VideoCapture(name)
    _, lastframe = Reader.read()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(name + '.mp4', fourcc, 30, (w, h))
    for index in range(0, 100, 2):
        if 'yuv' in name:
            IMAGE1, success1 = Reader.read(index)
            gt, _ = Reader.read(index + 1)
            IMAGE2, success2 = Reader.read(index + 2)
            if not success2:
                break
        else:
            success1, gt = Reader.read()
            success2, frame = Reader.read()
            IMAGE1 = lastframe
            IMAGE2 = frame
            lastframe = frame
            if not success2:
                break

        # BGR -> RBG
        IMAGE1 = IMAGE1[:, :, ::-1].copy()
        IMAGE2 = IMAGE2[:, :, ::-1].copy()
        gt = gt[:, :, ::-1].copy()

        I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        
        if h == 720:
            pad = 24
        elif h == 1080:
            pad = 4
        else:
            pad = 16
        pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
        I0 = pader(I0)
        I1 = pader(I1)

        # inference
        with torch.no_grad():
            pred = model.forward(I0, I1)["I_t"]
            pred = pred[:, :, pad: -pad]

            if TTA:
                I0_flip = I0.flip(2).flip(3)
                I1_flip = I1.flip(2).flip(3)
                pred_flip = model.forward(I0_flip, I1_flip)["I_t"]
                pred_flip = pred_flip[:, :, pad: -pad]
                pred = (pred + pred_flip.flip(2).flip(3)) / 2

        out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')

        # video.write(out)
        if 'yuv' in name:
            diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
            mse = np.mean((diff_rgb - 128.0) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        else:
            psnr = skim.compare_psnr(gt, out)
        psnr_list.append(psnr)
    print(f'{data[0]} ({data[0]}x{data[1]}), {np.mean(psnr_list)}')
    tot += np.mean(psnr_list)

print('avg psnr', tot / len(name_list))