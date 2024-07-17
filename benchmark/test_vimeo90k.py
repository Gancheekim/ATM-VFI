import cv2
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import warnings
import time
from pytorch_msssim import ssim_matlab
from utils import save_prediction
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
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/vimeo_triplet")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--TTA", action='store_true', default=False)
parser.add_argument("--TTA_swaporder", action='store_true', default=False)
parser.add_argument("--save_fig", action='store_true', default=False)
parser.add_argument("--profiling", action='store_true', default=False)
parser.add_argument("--ckpt", type=str, default="../../research3_ckpt/ours-final/vimeo_epoch_254_psnr_36.3847.pt") # network_base (final)
# parser.add_argument("--ckpt", type=str, default="../finetune_ckpt66/vimeo_epoch_274_psnr_35.9854.pt") # network_lite
# parser.add_argument("--ckpt", type=str, default="../finetune_ckpt81/vimeo_epoch_98_psnr_35.9552.pt") # network_base (final-perception-new)

args = parser.parse_args()

model = Network()
load_model_checkpoint(model, args.ckpt)
device = torch.device(args.device)
model.to(device).eval()

model.global_motion = False # only for Network55
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")    


TTA = args.TTA
save_fig = args.save_fig
print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K\t TTA: {args.TTA}\t TTA using swap frame order: {args.TTA_swaporder}')
path = args.path
f = open(path + '/tri_testlist.txt', 'r')
psnr_list, ssim_list = [], []
for counter, i in enumerate(f):
	if not args.profiling:
		print(f'{counter+1}', end='\r')
	name = str(i).strip()
	if(len(name) <= 1):
		continue
	I0 = cv2.imread(path + '/sequences/' + name + '/im1.png')
	I1 = cv2.imread(path + '/sequences/' + name + '/im2.png')
	I2 = cv2.imread(path + '/sequences/' + name + '/im3.png') 
	
	# BGR -> RBG
	I0 = I0[:, :, ::-1].copy()
	I1 = I1[:, :, ::-1].copy()
	I2 = I2[:, :, ::-1].copy()

	I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
	I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0) # normalized from 0-255 to 0-1

	mid = model.forward(I0, I2)['I_t'][0]

	if TTA:
		I0_flip = I0.flip(2).flip(3)
		I2_flip = I2.flip(2).flip(3)
		mid_flip = model.inference(I0_flip, I2_flip)[0]
		mid_TTA = mid_flip.flip(1).flip(2)
		mid = (mid + mid_TTA) / 2

	if args.TTA_swaporder:
		mid_SA = model.inference(I2, I0)[0]
		mid_SA_flip = model.inference(I2_flip, I0_flip)[0]
		mid_SA_flip = mid_SA_flip.flip(1).flip(2)
		mid_SA = (mid_SA + mid_SA_flip) / 2
		mid = (mid + mid_SA) / 2

	if save_fig:
		mid_ = mid.clone().unsqueeze(0)
	
	# evaluate
	ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
	mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
	I1 = I1 / 255.
	psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
	psnr_list.append(psnr)
	ssim_list.append(ssim)

	if save_fig:
		save_prediction(im1=I0, 
				  		im3=I2, 
						im2_pred=mid_,
						im2_label=(torch.tensor(I1.transpose(2, 0, 1)).to(device)).unsqueeze(0), 
						psnr=[psnr],
						visualize_idx=counter, 
						visualization_path="./vimeo90k_network26"
						)

print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
