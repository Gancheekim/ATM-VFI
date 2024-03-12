import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

from pytorch_msssim import ssim_matlab
sys.path.append('../')

""" different network """
# from network6 import Network
# from network18 import Network
# from network19 import Network
from network22 import Network


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
	

def load_model_checkpoint(model, checkpoint_path):
	param = torch.load(checkpoint_path, map_location='cuda:0')
	layers_to_remove = []
	for key in param:
		if "pretrained_flow_net" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.attn_mask" in key:
			layers_to_remove.append(key)
		if "translation_predictor.1.HW" in key:
			layers_to_remove.append(key)
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/middlebury/")
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 1.91, TTA: 
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 1.88
args = parser.parse_args()

myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
load_model_checkpoint(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")    

TTA = args.TTA

print(f'=========================Starting testing=========================')
print(f'Dataset: MiddleBury\t TTA: {TTA}')
path = args.path
name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
for i in name:
	i0 = cv2.imread(path + '/other-data/{}/frame10.png'.format(i))
	i1 = cv2.imread(path + '/other-data/{}/frame11.png'.format(i))
	gt = cv2.imread(path + '/other-gt-interp/{}/frame10i11.png'.format(i)) 

	# BGR -> RBG
	i0 = i0[:, :, ::-1].copy().transpose(2, 0, 1) / 255.
	i1 = i1[:, :, ::-1].copy().transpose(2, 0, 1) / 255.
	gt = gt[:, :, ::-1].copy()

	i0 = torch.from_numpy(i0).unsqueeze(0).float().cuda()
	i1 = torch.from_numpy(i1).unsqueeze(0).float().cuda()
	padder = InputPadder(i0.shape, divisor = 32)
	i0, i1 = padder.pad(i0, i1)
		
	with torch.no_grad():
		pred1 = model.forward(i0, i1)["I_t"][0] # inference

	if TTA:
		i0 = i0.flip(2).flip(3)
		i1 = i1.flip(2).flip(3)
		with torch.no_grad():
			pred1_flip = model.forward(i0, i1)["I_t"][0]
		pred1 = (pred1 + pred1_flip.flip(1).flip(2)) / 2

	pred = padder.unpad(pred1)
	out = pred.detach().cpu().numpy().transpose(1, 2, 0)
	out = np.round(out * 255.)
	IE_list.append(np.abs((out - gt * 1.0)).mean())


print(f"Avg IE: {np.mean(IE_list)}")