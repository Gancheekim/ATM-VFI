import os
import sys
import cv2
import math
import glob
import torch
import argparse
import warnings
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

'''==========import from our code=========='''
sys.path.append('.')
from utils import InputPadder
from pytorch_msssim import ssim_matlab

# from network37 import Network
# from network37_large import Network
# from network38 import Network
# from network39 import Network
# from network40 import Network
# from network41 import Network
# from network44 import Network
# from network53 import Network
from network55 import Network

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
		if ".relative_coord" in key:
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


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/home/kim/Desktop/ssd/X4K1000FPS/test')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--TTA", action='store_true', default=False)
parser.add_argument("--TTA_swaporder", action='store_true', default=False)
parser.add_argument("--save_fig", action='store_true', default=False)
parser.add_argument("--profiling", action='store_true', default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 34.44/0.9737
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 35.28/0.9771
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 35.965/0.9797, TTA: 36.117/0.9802
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 36.227/0.9806
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints17/epoch_295_psnr_36.3969.pt") # network26 -> 36.396/0.9814 , TTA: 36.54/0.9818
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints24/epoch_280_psnr_36.4069.pt") # network37
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints27/epoch_85_x4k_psnr_25.2543.pt") # network40
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints25/epoch_188_psnr_27.8384.pt") # network37_large
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints34/x4k_epoch_275psnr_26.5779.pt") # network44
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints34/vimeo_epoch_272psnr_36.219.pt") # network44 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints49/epoch_114_psnr_26.5911.pt") # network53
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints51/epoch_193_psnr_25.3407.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_247_psnr_31.2068.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/vimeo_epoch_246_psnr_36.3443.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/vimeo_epoch_270_psnr_36.3497.pt") # network55
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_275_psnr_27.294.pt") # network55

'''
network37:
- 2K: 27.22/0.8101
- 4K: OOM

network37_large:
- 2K: 27.44/0.8642
- 4K: 29.65/0.8907

network40:
- 2K: 29.54/0.8888
- 4K: 28.20/0.8738

network44 (epoch 275):
- 2K: 31.15/0.9062
- 4K (down 2): 30.19/0.9018

network44 (epoch 272):
- 2K: 31.12/0.906
- 4K (down 2): 30.02/0.8999

network53 (epoch 114):
- 2K: 28.97/0.8602
- 4K (down 2): 28.10/0.8623

network55, global=True (epoch 247):
- 2K: 30.41/0.8941
- 4K (down 2): 29.57/0.8927

network55, global=False (epoch 247):
- 2K: 27.67/0.8181
- 4K (down 2): 26.90/0.8321

network55, global=True (epoch 246):
- 2K: 30.15/0.8911
- 4K (down 2): 29.29/0.8901

network55, global=False (epoch 246):
- 2K: 27.60/0.8150
- 4K (down 2): 26.81/0.8299

network55, global=True (epoch 270):
- 2K: 30.07/0.8877
- 4K (down 2): 29.19/0.8875

network55, global=True, ensemble global (epoch 270):
- 2K: 30.67/0.9005
- 4K (down 2): 29.79/0.8975

network55, global=True (epoch 275):
- 2K: 30.59/0.8972
- 4K (down 2): 29.71/0.8950
'''
args = parser.parse_args()

myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
# load_model_checkpoint(model, args.model_checkpoints, device='cuda:0' if args.device == 'cuda' else 'cpu')
load_model_checkpoint1(model, args.model_checkpoints)
# checkpts = glob.glob(os.path.join("../finetune_model_checkpoints52/", "x4k_epoch" + '*.pt'))
# checkpts = glob.glob(os.path.join("../finetune_model_checkpoints52/", "vimeo_epoch" + '*.pt'))

for cp in [args.model_checkpoints]:
# for cp in checkpts:
	print()
	print(cp)
	model = Network()
	load_model_checkpoint1(model, cp, log_meta=False)

	device = torch.device(args.device)
	model.to(device).eval()

	model.global_motion = True # only for Network55
	# model.global_motion = False # only for Network55

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")    

	def getXVFI(dir, multiple=2, t_step_size=32):
		""" make [I0,I1,It,t,scene_folder] """
		testPath = []
		t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
		for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):
			for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):
				frame_folder = sorted(glob.glob(scene_folder + '*.png'))
				for idx in range(0, len(frame_folder), t_step_size):
					if idx == len(frame_folder) - 1:
						break
					for mul in range(multiple - 1):
						I0I1It_paths = []
						I0I1It_paths.append(frame_folder[idx])
						I0I1It_paths.append(frame_folder[idx + t_step_size])
						I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])
						I0I1It_paths.append(t[mul])
						testPath.append(I0I1It_paths)

		return testPath


	TTA = args.TTA
	print(f'=========================Starting testing=========================')
	print(f'Dataset: X4K1000FPS\t TTA: {TTA}')
	data_path = args.path
	listFiles = getXVFI(data_path)
	for strMode in ['XTEST-4k', 'XTEST-2k']:
	# for strMode in ['XTEST-2k']:
		fltPsnr, fltSsim = [], []
		for intFrame in tqdm(listFiles):
			npyOne = np.array(cv2.imread(intFrame[0])).astype(np.float32) * (1.0 / 255.0)
			npyTwo = np.array(cv2.imread(intFrame[1])).astype(np.float32) * (1.0 / 255.0)
			npyTruth = np.array(cv2.imread(intFrame[2])).astype(np.float32) * (1.0 / 255.0)
			# BGR->RGB
			npyOne = npyOne[:, :, ::-1]
			npyTwo = npyTwo[:, :, ::-1]
			npyTruth = npyTruth[:, :, ::-1]

			if strMode == 'XTEST-2k': # downscale
				npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
				npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
				npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

			elif strMode == 'XTEST-4k': # center-crop
				# mode = 'center-crop'
				mode = 'downscale-2'
				if mode == 'center-crop':
					H, W, _ = npyOne.shape
					npyOne = npyOne[270:-270, 512:-512, :]
					npyTwo = npyTwo[270:-270, 512:-512, :]
					npyTruth = npyTruth[270:-270, 512:-512, :]

			tenOne = torch.FloatTensor(np.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
			tenTwo = torch.FloatTensor(np.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
			tenGT = torch.FloatTensor(np.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()

			if strMode == 'XTEST-4k' and mode == 'downscale-2':
				downscale = 2
				tenOne = F.interpolate(tenOne, scale_factor=1/downscale, mode='bilinear', align_corners=True)
				tenTwo = F.interpolate(tenTwo, scale_factor=1/downscale, mode='bilinear', align_corners=True)

			padder = InputPadder(tenOne.shape, 32)
			tenOne, tenTwo = padder.pad(tenOne, tenTwo)
			
			with torch.no_grad():
				tenEstimate = model.forward(tenOne, tenTwo)['I_t'][0]
				# tenEstimate = model.inference(tenOne, tenTwo)[0]
	
				if TTA:
					tenOne_flip = tenOne.flip(2).flip(3)
					tenTwo_flip = tenTwo.flip(2).flip(3)
					tenEstimate_flip = model.forward(tenOne_flip, tenTwo_flip)['I_t'][0]
					tenEstimate_TTA = tenEstimate_flip.flip(1).flip(2)
					tenEstimate = (tenEstimate + tenEstimate_TTA) / 2

				tenEstimate = padder.unpad(tenEstimate)

				if strMode == 'XTEST-4k' and mode == 'downscale-2':
					tenEstimate = F.interpolate(tenEstimate[None], scale_factor=downscale, mode='bicubic', align_corners=True)[0]

			npyEstimate = (tenEstimate.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(np.uint8)
			tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

			fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
			fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())

		print(f'{strMode}  PSNR: {np.mean(fltPsnr)}  SSIM: {np.mean(fltSsim)}')