import os
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import warnings
import sys
import torch.nn.functional as F
from pytorch_msssim import ssim_matlab

sys.path.append('../')
""" different network """
from network_base import Network
# from network_lite import Network

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)


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

     

def load_model_checkpoint(model, checkpoint_path, strict=True):
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
parser.add_argument("--model_checkpoints", type=str, default="../../research3_ckpt/ours-final/vimeo_epoch_254_psnr_36.3847.pt") # network_base (final)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints66/vimeo_epoch_274_psnr_35.9854.pt") # network_lite
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints81/vimeo_epoch_98_psnr_35.9552.pt") # network_base (final-perception-new)

args = parser.parse_args()

model = Network()
load_model_checkpoint(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# set model run-time hyperparameters:
model.global_motion = True
model.ensemble_global_motion = False
# model.__set_global_window_size__(window_size=8)

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
		padder = InputPadder(I0.shape, divisor=64)
		I0, I2 = padder.pad(I0, I2)

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
