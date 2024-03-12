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

from pytorch_msssim import ssim_matlab
sys.path.append('../')

""" different network """
# from network6 import Network
# from network18 import Network
# from network19 import Network
from network22 import Network

import torch.nn.functional as F
class InputPadder:
	""" Pads images such that dimensions are divisible by divisor """
	def __init__(self, dims, divisor = 16):
		self.ht, self.wd = dims[-2:]
		pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
		pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
		self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

	def pad(self, *inputs):
		return [F.pad(x, self._pad, mode='replicate') for x in inputs]

	def unpad(self,x):
		ht, wd = x.shape[-2:]
		c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
		return x[..., c[0]:c[1], c[2]:c[3]]

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
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/snufilm-test/eval_modes/")
parser.add_argument("--img_data_path", type=str, default="/home/kim/Desktop/ssd/snufilm-test/")
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> , TTA: 
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints17/epoch_295_psnr_36.3969.pt") # network26
"""
network22:
- easy: 40.368/0.9910
- medium: 36.094/0.9797
- hard: 30.367/0.9344
- extreme: 24.922/0.8561

network26:
- easy: 
- medium: 
- hard: 
- extreme: 
"""

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

TTA = args.TTA
print(f'=========================Starting testing=========================')
print(f'Dataset: SNU_FILM\t     TTA: {TTA}')
path = args.path
level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
for test_file in level_list:
	psnr_list, ssim_list = [], []
	file_list = []
	
	with open(os.path.join(path, test_file), "r") as f:
		for line in f:
			line = line.replace("data/SNU-FILM/test/", args.img_data_path)
			line = line.strip()
			file_list.append(line.split(' '))

	for line in tqdm(file_list):
		I0_path = os.path.join(path, line[0])
		I1_path = os.path.join(path, line[1])
		I2_path = os.path.join(path, line[2])
		I0 = cv2.imread(I0_path)
		I1_ = cv2.imread(I1_path)
		I2 = cv2.imread(I2_path)
		
		# BGR -> RBG
		I0 = I0[:, :, ::-1].copy()
		I1_ = I1_[:, :, ::-1].copy()
		I2 = I2[:, :, ::-1].copy()

		I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
		I1 = (torch.tensor(I1_.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
		I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
		padder = InputPadder(I0.shape, divisor=32)
		I0, I2 = padder.pad(I0, I2)
		
		I1_pred = model.forward(I0, I2)["I_t"][0] # inference

		if TTA:
			I0_flip = I0.flip(2).flip(3)
			I2_flip = I2.flip(2).flip(3)
			I1_pred_flip = model.forward(I0_flip, I2_flip)["I_t"][0]
			I1_pred = (I1_pred + I1_pred_flip.flip(1).flip(2)) / 2
			
		I1_pred = padder.unpad(I1_pred)
		
		ssim = ssim_matlab(I1, I1_pred.unsqueeze(0)).detach().cpu().numpy()

		I1_pred = I1_pred.detach().cpu().numpy().transpose(1, 2, 0)   
		I1_ = I1_ / 255.
		psnr = -10 * math.log10(((I1_ - I1_pred) * (I1_ - I1_pred)).mean())
		
		psnr_list.append(psnr)
		ssim_list.append(ssim)
	
	print('Testing level:' + test_file[:-4])
	print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
