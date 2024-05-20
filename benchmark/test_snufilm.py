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
# from network22 import Network
# from network37 import Network
# from network44 import Network
# from network55 import Network
# from network57 import Network
from network57_small2 import Network

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

def load_model_checkpoint(model, checkpoint_path, device):
	param = torch.load(checkpoint_path, map_location=device)
	layers_to_remove = []
	for key in param:
		if "pretrained_flow_net" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.HW" in key:
			layers_to_remove.append(key)

		if "bottleneck.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "bottleneck.1.HW" in key:
			layers_to_remove.append(key)
		
		if "translation_predictor.0.attn.relative_coord_x" in key:
			layers_to_remove.append(key)
		if "translation_predictor.0.attn.relative_coord_y" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.attn.relative_coord_x" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.attn.relative_coord_y" in key:
			layers_to_remove.append(key)

	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param)

	# model.translation_predictor[0].attn.register_relative_coord()
	# model.translation_predictor[1].attn.register_relative_coord()
	model.translation_predictor[0].attn._register_relative_coord_()
	model.translation_predictor[1].attn._register_relative_coord_()

def load_model_checkpoint1(model, checkpoint_path, strict=False, log_meta=True):
	checkpt = torch.load(checkpoint_path, map_location='cuda:0')
	try:
		param = checkpt['model_state_dict']
		optimizer = checkpt['optimizer_state_dict']
		meta_data = checkpt['meta_data']
		train_metric = checkpt['train_metric']
		val_metric = checkpt['val_metric']
		if log_meta:
			print(meta_data)
			print(f'\t- train: {train_metric}\n\t- val: {val_metric}')
	except:
		param = checkpt

	layers_to_remove = []
	for key in param:
		if "relative_coord" in key:
			layers_to_remove.append(key)

		if "local_motion_transformer.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "local_motion_transformer.1.HW" in key:
			layers_to_remove.append(key)
		if "global_motion_transformer.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "global_motion_transformer.1.HW" in key:
			layers_to_remove.append(key)
			
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param, strict=strict)

	model.local_motion_transformer[0].attn._register_relative_coord_()     
	model.local_motion_transformer[1].attn._register_relative_coord_()     
	model.global_motion_transformer[0].attn._register_relative_coord_()     
	model.global_motion_transformer[1].attn._register_relative_coord_()       

def load_model_checkpoint2(model, checkpoint_path, strict=True):
	print(f'--- loading from checkpoint: {checkpoint_path} ---')
	checkpt = torch.load(checkpoint_path, map_location='cuda:0')
	try:
		param = checkpt['model_state_dict']
		optim_checkpt = checkpt['optimizer_state_dict']
		meta_data = checkpt['meta_data']
		train_metric = checkpt['train_metric']
		val_metric = checkpt['val_metric']
		print(meta_data)
		print(f'\t- train: {train_metric}\n\t- val: {val_metric}')
	except:
		param = checkpt

	layers_to_remove = []
	for key in param:
		# if "relative_coord" in key:
			# layers_to_remove.append(key)
		if "attn_mask" in key:
			layers_to_remove.append(key)
		elif "HW" in key:
			layers_to_remove.append(key)
			
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param, strict=strict)

	return optim_checkpt



parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/snufilm-test/eval_modes/")
parser.add_argument("--img_data_path", type=str, default="/home/kim/Desktop/ssd/snufilm-test/")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> , TTA: 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints17/epoch_295_psnr_36.3969.pt") # network26
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints24/epoch_280_psnr_36.4069.pt") # network37 -> / , TTA: /
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints34/x4k_epoch_275psnr_26.5779.pt") # network44
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_247_psnr_31.2068.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_275_psnr_27.294.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/vimeo_epoch_270_psnr_36.3497.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints53/x4k_large_epoch_80_psnr_25.6615.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints55/vimeo_epoch_88_psnr_36.3716.pt") # network55

# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_254_psnr_36.3847.pt") # network57: 30.62(hard)

# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/x4k_snuEx_epoch_221_psnr_30.6193.pt") # network57
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/x4k_snuEx_epoch_169_psnr_30.6108.pt") # network57
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_238_psnr_36.3775.pt") # network57

# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_270_psnr_36.3756.pt") # network57: 30.60(hard)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_268_psnr_36.3774.pt") # network57: 30.61
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_266_psnr_36.3768.pt") # network57: 30.60
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_264_psnr_36.3731.pt") # network57: 30.59
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_262_psnr_36.3718.pt") # network57: 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_260_psnr_36.3657.pt") # network57: 

# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_220_psnr_36.3698.pt") # network57: 30.57(hard)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_222_psnr_36.3594.pt") # network57: 30.61

# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_256_psnr_36.3843.pt") # network57: 30.62/30.60
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_224_psnr_36.3823.pt") # network57: 30.60

parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints66/vimeo_epoch_274_psnr_35.9854.pt") # network57_small2

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

network37: (TTA=False)
- easy: 40.18/0.9910
- medium: 36.03/0.9797
- hard: 30.39/0.9350
- extreme: 24.93/0.8566

network37: (TTA=True)
- easy: 40.26/0.9910
- medium: 36.12/0.9799
- hard: 30.48/0.9356
- extreme: 25.03/0.8580

network44: (TTA=False)
- easy: 40.42/0.9911
- medium: 36.14/0.9797
- hard: 30.49/0.9351
- extreme: 24.99/0.8561

network55(epoch 247): (TTA=False), global window=12
- easy: 40.30/0.9909
- medium: 36.09/0.9795
- hard: 30.70/0.9374
- extreme: 25.60/0.8676

network55(epoch 247): (TTA=False), global window=14
- easy: 40.32/0.9910
- medium: 36.12/0.9796
- hard: 30.74/0.9376
- extreme: 25.59/0.8676

network55(epoch 247): (TTA=False), global window=15
- easy: 40.33/0.9910
- medium: 36.12/0.9796
- hard: 30.73/0.9375
- extreme: 25.57/0.8674

network55(epoch 247): (TTA=False), global window=10
- easy: 40.32/0.9910
- medium: 36.12/0.9796
- hard: 30.71/0.9375
- extreme: 25.60/0.8678

network55(epoch 275): (TTA=False), global window=12
- easy: 40.26/0.9908
- medium: 36.08/0.9794
- hard: 30.72/0.9376
- extreme: 25.63/0.8685

network55(epoch 270): (TTA=False), global window=12
- easy: 40.25/0.9910
- medium: 36.04/0.9796
- hard: 30.60/0.9367
- extreme: 25.43/0.8648

network55(epoch 270): (TTA=True), global window=12, ensemble global
- easy: 40.27/0.9908
- medium: 36.05/0.9794
- hard: 30.70/0.9376
- extreme: 25.60/0.8685

network55(epoch 80): (TTA=False), global window=12
- easy: 40.13/0.9906
- medium: 35.99/0.9794
- hard: 30.70/0.9376
- extreme: 25.66/0.8692

network55(epoch 80): (TTA=False), global window=12, ensemble global
- easy: 40.08/0.9905
- medium: 35.98/0.9791
- hard: 30.68/0.9376
- extreme: 25.65/0.8693

network55(epoch 88): (TTA=False), global window=12, ensemble global
- easy: 40.21/0.9910
- medium: 36.01/0.9795
- hard: 30.52/0.9363
- extreme: 25.37/0.8647

network55(epoch 88): (TTA=False), global window=12
- easy: 40.23/0.9910, global off
- medium: 36.05/0.9797, global off
- hard: 30.51/0.9361, global on (win=8)
- extreme: 25.40/0.8645, global on (win=8), ensemble

network57(epoch 254): (TTA=False), default global window=12
- easy: 40.24/0.9910  	 global off | 40.23/0.9910 global on
- medium: 36.09/0.9798   global off | 36.08/0.9798 global on
- hard: 30.62/0.9369, 	 global on  | 30.60/0.9369 ensemble global
- extreme: 25.48/0.8658, global on  | 25.46/0.8659 ensemble global

network57(epoch 254): (TTA=True), default global window=8
- hard: 30.74/0.9375, 	 global on  | 30.74/0.9375 ensemble global
- extreme: 25.60/0.8676, global on  | 25.61/0.8680 ensemble global | 25.21 global off

network57(epoch 254): (TTA=True), default global window=12
- hard: 30.72/0.9374, 	 global on  |  ensemble global
- extreme: 25.58/0.8673, global on  |  ensemble global

network57(epoch 221): (TTA=False), default global window=12
- easy: 40.30/0.9909  	 global off | 40.30/0.9909 global on
- medium: 36.13/0.9796   global off | 36.13/0.9796 global on
- hard: 30.76/0.9380, 	 global on  | 30.75/0.9379 ensemble global
- extreme: 25.65/0.8685, global on  | 25.67/0.8690 ensemble global

network57(epoch 238): (TTA=False), default global window=12
- hard: 30.61/0.9369, 	 global on  | 30.60/0.9369 ensemble global
- extreme: 25.47/0.8657, global on  | 25.47/0.8661 ensemble global

network57_small2(epoch 274): (TTA=False), default global window=12
- easy: 40.23/0.9909  	 global off | 40.23/0.9909 global on
- medium: 35.92/0.9795   global off | 35.91/0.9794 global on
- hard: 30.48/0.9356, 	 global on  | 30.46/0.9356 ensemble global(win=12) | 30.46/0.9355 ensemble global(win=8),
- extreme: 25.33/0.8629, global on  | 25.33/0.8630 ensemble global(win=12) | 25.33/0.8636 ensemble global(win=8),
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

# load_model_checkpoint(model, args.model_checkpoints, args.device)
load_model_checkpoint1(model, args.model_checkpoints)
# load_model_checkpoint2(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# model.global_motion = False

# model.ensemble_global_motion = False
model.ensemble_global_motion = True

model.__set_global_window_size__(window_size=8)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")    

TTA = args.TTA
print(f'=========================Starting testing=========================')
print(f'Dataset: SNU_FILM\t     TTA: {TTA}')
path = args.path
# level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
# level_list = ['test-easy.txt', 'test-medium.txt'] 
level_list = ['test-hard.txt', 'test-extreme.txt'] 
# level_list = ['test-hard.txt'] 
# level_list = ['test-extreme.txt'] 
for test_file in level_list:
	psnr_list, ssim_list = [], []
	file_list = []

	# if 'easy' in test_file:
	# 	model.global_motion = False
	# elif 'medium' in test_file:
	# 	model.global_motion = False
	# elif 'hard' in test_file:
	# 	model.global_motion = True
	# 	model.__set_global_window_size__(window_size=8)
	# elif 'extreme' in test_file:
	# 	model.global_motion = True
	# 	model.ensemble_global_motion = True
	# 	model.__set_global_window_size__(window_size=8)
	
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
		# padder = InputPadder(I0.shape, divisor=32)
		padder = InputPadder(I0.shape, divisor=64)
		I0, I2 = padder.pad(I0, I2)
		
		# I1_pred = model.inference(I0, I2)[0] # inference
		# print(I0.size())

		I1_pred = model.forward(I0, I2)['I_t'][0]

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
