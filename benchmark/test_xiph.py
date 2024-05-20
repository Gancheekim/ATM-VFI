import os
import sys
import cv2
import tqdm
import glob
import torch
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import torch.nn.functional as F

from utils import InputPadder, read, img2tensor
from psnr_ssim import calculate_psnr, calculate_ssim

""" different network """
sys.path.append('../')
# from network6 import Network
# from network18 import Network
# from network19 import Network
# from network22 import Network
# from network26 import Network
# from network37 import Network
# from network40 import Network
# from network44 import Network
# from network55 import Network
# from network57 import Network
from network57_small2 import Network

def load_model_checkpoint(model, checkpoint_path, device='cuda:0'):
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
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/amt-s.pth') 
parser.add_argument('-r', '--root', default='/home/kim/Desktop/ssd/xiph') 
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 34.44/0.9737
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 35.28/0.9771
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 35.965/0.9797, TTA: 36.117/0.9802
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 36.227/0.9806
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints17/epoch_112_psnr_36.1739.pt") # network26 -> , TTA: 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints24/epoch_280_psnr_36.4069.pt") # network37
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints27/epoch_85_x4k_psnr_25.2543.pt") # network40
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints34/x4k_epoch_275psnr_26.5779.pt") # network44
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_247_psnr_31.2068.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/x4k_epoch_275_psnr_27.294.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/vimeo_epoch_270_psnr_36.3497.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints55/vimeo_epoch_88_psnr_36.3716.pt") # network55
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_254_psnr_36.3847.pt") # network57

parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints66/vimeo_epoch_274_psnr_35.9854.pt") # network57_small2

'''
network37 (w/o TTA):
- 2k: 36.54/0.9675
- 4k: 33.31/0.9447

network40 (w/o TTA):
- 2k: 35.16/0.9618
- 4k: 

network44 (w/o TTA):
- 2k: 36.80/0.9676
- 4k: 34.20/0.9473

network55(epoch 247) (w/o TTA): global_win=12
- 2k: 36.58/0.9656
- 4k: 34.14/0.9471	

network55(epoch 247) (w/o TTA): global_win=14
- 2k: 36.59/0.9656
- 4k: 34.13/0.9471	

network55(epoch 247) (w/o TTA): global_win=16
- 2k: 36.59/0.9656
- 4k: 34.13/0.9471

network55(epoch 247) (w/o TTA): global_win=10
- 2k: 36.55/0.9656
- 4k: 34.14/0.9471	

network55(epoch 275) (w/o TTA): global_win=12
- 2k: 36.53/0.9651
- 4k: 34.14/0.9469

network55(epoch 270) (w/o TTA): global_win=12
- 2k: 36.89/0.9673
- 4k: 34.19/0.9474

network55(epoch 270) (w/o TTA): global_win=12
- 2k: 36.84/0.9669
- 4k: 34.10/0.9472

network57(epoch 254) (w/o TTA): global_win=12
- 2k: 36.91/0.9677
- 4k: 34.15/0.9475, 34.29/0.9480 (TTA=True)

network57_small2(epoch 274) (w/o TTA): global_win=12
- 2k: 36.91/0.9677
- 4k: 33.92/0.9462,  (TTA=True)
'''
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
root = args.root
TTA = args.TTA

myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
# model.__set_global_window_size__(window_size=14)

# load_model_checkpoint(model, args.model_checkpoints)
# load_model_checkpoint1(model, args.model_checkpoints)
load_model_checkpoint2(model, args.model_checkpoints)
model.to(device).eval()

model.global_motion = True
# model.ensemble_global_motion = True
# model.__set_global_window_size__(window_size=16)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")

print("prepare to download...")
############################################# Prepare Dataset #############################################
download_links = [
	'https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m',
	'https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m',
]
file_list = ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 
			 'SquareAndTimelapse', 'Tango']

for file_name, link in zip(file_list, download_links):
	data_dir = osp.join(args.root, file_name)
	if osp.exists(data_dir) is False:
		os.makedirs(data_dir)
	if len(glob.glob(f'{data_dir}/*.png')) < 100:
		os.system(f'ffmpeg -i {link} -pix_fmt rgb24 -vframes 100 {data_dir}/%03d.png')
############################################### Prepare End ###############################################
print("prepare end.")

divisor = 32; scale_factor = 0.5
# for category in ['resized-2k', 'cropped-4k']:
# for category in ['cropped-4k']:
for category in ['resized-2k']:
	psnr_list = []
	ssim_list = []
	pbar = tqdm.tqdm(file_list, total=len(file_list))
	for flie_name in pbar:
		dir_name = osp.join(root, flie_name)
		for intFrame in range(2, 99, 2):
			img0 = read(f'{dir_name}/{intFrame - 1:03d}.png')
			img1 = read(f'{dir_name}/{intFrame + 1:03d}.png')
			imgt = read(f'{dir_name}/{intFrame:03d}.png')

			if category == 'resized-2k':
				img0 = cv2.resize(src=img0, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
				img1 = cv2.resize(src=img1, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
				imgt = cv2.resize(src=imgt, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

			elif category == 'cropped-4k':
				img0 = img0[540:-540, 1024:-1024, :]
				img1 = img1[540:-540, 1024:-1024, :]
				imgt = imgt[540:-540, 1024:-1024, :]
			img0 = img2tensor(img0).to(device)
			imgt = img2tensor(imgt).to(device)
			img1 = img2tensor(img1).to(device)
			
			padder = InputPadder(img0.shape, divisor)
			img0, img1 = padder.pad(img0, img1)

			# inference
			with torch.no_grad():
				imgt_pred = model.forward(img0, img1)["I_t"] # inference

				# img0 = F.interpolate(img0, scale_factor=0.5, mode='bilinear', align_corners=True)
				# img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=True)
				# imgt_pred = model.forward(img0, img1)["I_t"] # inference
				# imgt_pred = F.interpolate(imgt_pred, scale_factor=2, mode='bilinear', align_corners=True)

				if TTA:
					img0_flip = img0.flip(2).flip(3)
					img1_flip = img1.flip(2).flip(3)
					imgt_pred_flip = model.forward(img0_flip, img1_flip)["I_t"]
					imgt_pred = (imgt_pred + imgt_pred_flip.flip(2).flip(3)) / 2

				imgt_pred = padder.unpad(imgt_pred)

			psnr = calculate_psnr(imgt_pred, imgt)
			ssim = calculate_ssim(imgt_pred, imgt)

			avg_psnr = np.mean(psnr_list)
			avg_ssim = np.mean(ssim_list)
			psnr_list.append(psnr)
			ssim_list.append(ssim)
			desc_str = f'[{"network26"}/Xiph] [{category}/{flie_name}] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'

			pbar.set_description_str(desc_str)