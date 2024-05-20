import cv2
import argparse
import torch
import numpy as np
import os
import glob
import sys
sys.path.append('../')

from network57 import Network

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
parser.add_argument("--TTA", type=bool, default=False)
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints62/vimeo_epoch_254_psnr_36.3847.pt") # network57
parser.add_argument("--path", type=str, default="/home/kim/ssd/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/JPEGImages/480p/")
parser.add_argument("--id", type=str, default="breakdance-flare")

args = parser.parse_args()
myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)
	 

model = Network()
load_model_checkpoint2(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# model.global_motion = True
# model.global_motion = False

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M") 

frames = glob.glob(os.path.join(args.path + '/' + args.id, '*.jpg'))
frames.sort()

time_interval = 2
H, W = 480, 832
FPS = 10.0
# FPS = 5.0
INTERPOLATE4X = True
# INTERPOLATE4X = False

print(f'id: {args.id}')
print(f'time interval: {time_interval}')
print(f'fps: {FPS}')
print(f'INTERPOLATE4X: {INTERPOLATE4X}')
print(f'TTA: {args.TTA}')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outVid = cv2.VideoWriter('./video/output.mp4', fourcc, FPS, (W,H))

for i in range(0, len(frames)-time_interval, time_interval):
	frame1 = frames[i]
	frame2 = frames[i+time_interval]

	img0_np = cv2.imread(frame1)
	img1_np = cv2.imread(frame2)
	H_, W_, _ = img0_np.shape
	img0 = img0_np[H_//2-H//2:H_//2+H//2 , W_//2-W//2:W_//2+W//2, ::-1].copy()
	img1 = img1_np[H_//2-H//2:H_//2+H//2 , W_//2-W//2:W_//2+W//2, ::-1].copy()

	img0 = (torch.tensor(img0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
	img1 = (torch.tensor(img1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0) 

	with torch.no_grad():
		pred = model.forward(img0, img1)["I_t"][0] 

		if INTERPOLATE4X:
			pred025 = model.forward(img0, pred.unsqueeze(0))["I_t"][0] 
			pred075 = model.forward(pred.unsqueeze(0), img1)["I_t"][0] 

		if args.TTA:
			I0_flip = img0.flip(2).flip(3)
			I2_flip = img1.flip(2).flip(3)
			pred_flip = model.forward(I0_flip, I2_flip)["I_t"][0]
			pred = (pred + pred_flip.flip(1).flip(2)) / 2

	pred = pred.detach().cpu().numpy().transpose(1,2,0)
	pred = np.round(pred * 255).astype(np.uint8)
	pred = pred[:, :, ::-1].copy() #rgb->bgr

	# cv2.imwrite(f"./video/{i}.jpg", pred)

	outVid.write(img0_np)
	if INTERPOLATE4X:
		pred025 = pred025.detach().cpu().numpy().transpose(1,2,0)
		pred025 = np.round(pred025 * 255).astype(np.uint8)
		pred025 = pred025[:, :, ::-1].copy() #rgb->bgr
		outVid.write(pred025)

	outVid.write(pred)

	if INTERPOLATE4X:
		pred075 = pred075.detach().cpu().numpy().transpose(1,2,0)
		pred075 = np.round(pred075 * 255).astype(np.uint8)
		pred075 = pred075[:, :, ::-1].copy() #rgb->bgr
		outVid.write(pred075)

outVid.write(img1_np)
outVid.release()
