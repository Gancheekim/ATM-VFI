import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

sys.path.append('../')
from pytorch_msssim import ssim_matlab

""" different network """
# from network6 import Network
# from network18 import Network
# from network19 import Network
from network22 import Network

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
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/UCF101-triplet/ucf101_interp_ours")
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 35.38/0.9695, TTA: 35.438/0.9697
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 35.441/0.9699
args = parser.parse_args()

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

print(f'=========================Starting testing=========================')
print(f'Dataset: UCF101\t TTA: {args.TTA}')
path = args.path
dirs = os.listdir(path)
psnr_list, ssim_list = [], []
for d in tqdm(dirs):
	img0 = cv2.imread(path + '/' + d + '/frame_00.png')
	img1 = cv2.imread(path + '/' + d + '/frame_02.png')
	gt = cv2.imread(path + '/' + d + '/frame_01_gt.png')

	# BGR -> RBG
	img0 = img0[:, :, ::-1].copy()
	img1 = img1[:, :, ::-1].copy()
	gt = gt[:, :, ::-1].copy()

	img0 = (torch.tensor(img0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
	img1 = (torch.tensor(img1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0) 
	gt = (torch.tensor(gt.transpose(2, 0, 1) / 255.)).cuda().float().unsqueeze(0) # normalized from 0-255 to 0-1
		
	pred = model.forward(img0, img1)["I_t"][0] # inference
	
	if args.TTA:
		I0_flip = img0.flip(2).flip(3)
		I2_flip = img1.flip(2).flip(3)
		pred_flip = model.forward(I0_flip, I2_flip)["I_t"][0]
		pred = (pred + pred_flip.flip(1).flip(2)) / 2
	
	ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
	out = pred.detach().cpu().numpy().transpose(1, 2, 0)
	out = np.round(out * 255) / 255.
	gt = gt[0].cpu().numpy().transpose(1, 2, 0)
	psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
	psnr_list.append(psnr)
	ssim_list.append(ssim)
	
print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))