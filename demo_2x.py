import cv2
import torch
import argparse
import warnings
import numpy as np
import sys
from benchmark.utils import InputPadder

''' import model '''
sys.path.append('./network/')
from network_base import Network as Network_base
from network_lite import Network as Network_lite

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_type", type=str, default="base", choices=["base", "lite"])
	parser.add_argument("--model_checkpoints", type=str, default="../research3_ckpt/atm-vfi-base-pct.pt") # atm-vfi-base.pt/atm-vfi-lite.pt/atm-vfi-base-pct.pt
	parser.add_argument("--frame0", type=str, required=True)
	parser.add_argument("--frame1", type=str, required=True)
	parser.add_argument("--out", type=str, default="output_interpolation.png")
	args = parser.parse_args()

	if args.model_type == "base":
		model = Network_base()
	elif args.model_type == "lite":
		model = Network_lite()
	else:
		raise NotImplementedError

	load_model_checkpoint(model, args.model_checkpoints)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device).eval()
	# enable/disable global motion estimation, default=True
	model.global_motion = True

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")   

	# read input
	img0 = cv2.imread(args.frame0)
	img1 = cv2.imread(args.frame1)

	# BGR -> RBG
	img0 = img0[:, :, ::-1].copy()
	img1 = img1[:, :, ::-1].copy()

	# normalized from 0-255 to 0-1
	img0 = (torch.tensor(img0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
	img1 = (torch.tensor(img1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0) 
	
	# border padding
	padder = InputPadder(img0.shape, divisor=64)
	img0, img1 = padder.pad(img0, img1)

	# inference
	pred = model.forward(img0, img1)["I_t"][0] 
	pred = padder.unpad(pred)
	pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
	pred = np.round(pred * 255).astype(np.uint8)

	# RGB -> BGR
	pred = pred[:,:,::-1].copy()
	cv2.imwrite(args.out, pred)