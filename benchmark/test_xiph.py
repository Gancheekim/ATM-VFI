import os
import sys
import cv2
import tqdm
import glob
import torch
import argparse
import numpy as np
import os.path as osp
from utils import InputPadder, read, img2tensor
from psnr_ssim import calculate_psnr, calculate_ssim

sys.path.append('../')
""" different network """
from network_base import Network
# from network57_small2 import Network

myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)


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
parser.add_argument("--model_checkpoints", type=str, default="../../research3_ckpt/ours-final/vimeo_epoch_254_psnr_36.3847.pt") # network57: 30.62(hard) (final)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints66/vimeo_epoch_274_psnr_35.9854.pt") # network57_small2
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints81/vimeo_epoch_98_psnr_35.9552.pt") # network57 (final-perception-new)

args = parser.parse_args()
root = args.root
TTA = args.TTA

model = Network()
load_model_checkpoint2(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# set model run-time hyperparameters:
model.global_motion = True
# model.ensemble_global_motion = True
# model.__set_global_window_size__(window_size=12)

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
for category in ['resized-2k', 'cropped-4k']:
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

			with torch.no_grad():
				imgt_pred = model.forward(img0, img1)["I_t"] # inference

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