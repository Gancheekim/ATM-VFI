import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
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
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/UCF101-triplet/ucf101_interp_ours")
parser.add_argument("--TTA", type=bool, default=False)
parser.add_argument("--model_checkpoints", type=str, default="../../research3_ckpt/ours-final/vimeo_epoch_254_psnr_36.3847.pt") # network_base (final)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints66/vimeo_epoch_274_psnr_35.9854.pt") # network_lite
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints81/vimeo_epoch_98_psnr_35.9552.pt") # network_base (final-perception-new)

args = parser.parse_args()

model = Network()
load_model_checkpoint(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

model.global_motion = False

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