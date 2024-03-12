import cv2
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm
import time
from torchsummary import summary
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

from pytorch_msssim import ssim_matlab
from utils import save_prediction
sys.path.append('../')
from trainer import Trainer

""" different network """
# from network6 import Network
# from network18 import Network
# from network19 import Network
# from network22 import Network
from network26 import Network
# from network27 import Network


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

	model.translation_predictor[0].attn.register_relative_coord()
	model.translation_predictor[1].attn.register_relative_coord()


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/vimeo_triplet")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--TTA", action='store_true', default=False)
parser.add_argument("--TTA_swaporder", action='store_true', default=False)
parser.add_argument("--save_fig", action='store_true', default=False)
parser.add_argument("--profiling", action='store_true', default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 34.44/0.9737
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 35.28/0.9771
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 35.965/0.9797, TTA: 36.117/0.9802
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 36.227/0.9806
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints17/epoch_295_psnr_36.3969.pt") # network26 -> 36.396/-/9814 , TTA: 36.54/0.9818
args = parser.parse_args()

myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
load_model_checkpoint(model, args.model_checkpoints, device='cuda:0' if args.device == 'cuda' else 'cpu')
device = torch.device(args.device)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")    
# summary(model, [(3,256,256), (3,256,256)], device=device)

# from fvcore.nn import FlopCountAnalysis
# flops = FlopCountAnalysis(model, (torch.rand((1,3,256,256)).to(device), torch.rand((1,3,256,256)).to(device)) )
# print(flops.total())

from deepspeed.profiling.flops_profiler import FlopsProfiler
prof = FlopsProfiler(model)


TTA = args.TTA
save_fig = args.save_fig
print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K\t TTA: {args.TTA}\t TTA using swap frame order: {args.TTA_swaporder}')
path = args.path
f = open(path + '/tri_testlist.txt', 'r')
psnr_list, ssim_list = [], []
# profile_step = 200
profile_step = -1
profile = []
flops_list = []
macs_list = []
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

	# I0 = torch.rand(1,3,720,1280).cuda()
	# I1 = torch.rand(1,3,720,1280).cuda()
	# I2 = torch.rand(1,3,720,1280).cuda()
	
	# mid = model.forward(I0, I2)["I_t"][0] # inference
	if args.profiling and counter > 100:
		torch.cuda.synchronize()
		start = time.time()
		# prof.start_profile()

	mid = model.inference(I0, I2)[0]

	if args.profiling and counter > 100:
		torch.cuda.synchronize()
		end = time.time()
		profile.append((end-start)*1000)

		# prof.stop_profile()
		# flops = prof.get_total_flops()
		# macs = prof.get_total_macs()
		# flops_list.append(flops)
		# macs_list.append(macs)
		# prof.end_profile()

	if profile_step != -1 and counter == profile_step + 100:
		break

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
	
	# '''
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
	# '''						


print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
if args.profiling:
	avg_run_time = sum(profile) / len(profile)
	print(f'avg run time: {sum(profile)/len(profile)} ms')
	# print(f'avg flops (number of floating-point operations ): {sum(flops_list)/(len(flops_list)*1e12)} T')
	# print(f'avg multiply-accumulate operations (MACs): {sum(macs_list)/(len(macs_list)*1e12)} T')