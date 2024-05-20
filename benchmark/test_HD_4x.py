import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

from pytorch_msssim import ssim_matlab
sys.path.append('../')

# from network19 import Network
# from network22 import Network
# from network37 import Network
from network55 import Network

import torch.nn.functional as F
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

class YUV_Read():
	def __init__(self, filepath, h, w, format='yuv420', toRGB=True):

		self.h = h
		self.w = w

		self.fp = open(filepath, 'rb')

		if format == 'yuv420':
			self.frame_length = int(1.5 * h * w)
			self.Y_length = h * w
			self.Uv_length = int(0.25 * h * w)
		else:
			pass
		self.toRGB = toRGB

	def read(self, offset_frame=None):
		if not offset_frame == None:
			self.fp.seek(offset_frame * self.frame_length, 0)

		Y = np.fromfile(self.fp, np.uint8, count=self.Y_length)
		U = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
		V = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
		if Y.size < self.Y_length or \
						U.size < self.Uv_length or \
						V.size < self.Uv_length:
			return None, False

		Y = np.reshape(Y, [self.w, self.h], order='F')
		Y = np.transpose(Y)

		U = np.reshape(U, [int(self.w / 2), int(self.h / 2)], order='F')
		U = np.transpose(U)

		V = np.reshape(V, [int(self.w / 2), int(self.h / 2)], order='F')
		V = np.transpose(V)

		U = np.array(Image.fromarray(U).resize([self.w, self.h]))
		V = np.array(Image.fromarray(V).resize([self.w, self.h]))

		if self.toRGB:
			Y = Y / 255.0
			U = U / 255.0 - 0.5
			V = V / 255.0 - 0.5

			self.YUV = np.stack((Y, U, V), axis=-1)
			self.RGB = (255.0 * np.clip(yuv2rgb(self.YUV), 0.0, 1.0)).astype('uint8')

			self.YUV = None
			return self.RGB, True
		else:
			self.YUV = np.stack((Y, U, V), axis=-1)
			return self.YUV, True

	def close(self):
		self.fp.close()

class YUV_Write():
	def __init__(self, filepath, fromRGB=True):
		if os.path.exists(filepath):
			print(filepath)
  
		self.fp = open(filepath, 'wb')
		self.fromRGB = fromRGB

	def write(self, Frame):

		self.h = Frame.shape[0]
		self.w = Frame.shape[1]
		c = Frame.shape[2]

		assert c == 3
		if format == 'yuv420':
			self.frame_length = int(1.5 * self.h * self.w)
			self.Y_length = self.h * self.w
			self.Uv_length = int(0.25 * self.h * self.w)
		else:
			pass
		if self.fromRGB:
			Frame = Frame / 255.0
			YUV = rgb2yuv(Frame)
			Y, U, V = np.dsplit(YUV, 3)
			Y = Y[:, :, 0]
			U = U[:, :, 0]
			V = V[:, :, 0]
			U = np.clip(U + 0.5, 0.0, 1.0)
			V = np.clip(V + 0.5, 0.0, 1.0)

			U = U[::2, ::2]  # imresize(U,[int(self.h/2),int(self.w/2)],interp = 'nearest')
			V = V[::2, ::2]  # imresize(V ,[int(self.h/2),int(self.w/2)],interp = 'nearest')
			Y = (255.0 * Y).astype('uint8')
			U = (255.0 * U).astype('uint8')
			V = (255.0 * V).astype('uint8')
		else:
			YUV = Frame
			Y = YUV[:, :, 0]
			U = YUV[::2, ::2, 1]
			V = YUV[::2, ::2, 2]

		Y = Y.flatten()  # the first order is 0-dimension so don't need to transpose before flatten
		U = U.flatten()
		V = V.flatten()

		Y.tofile(self.fp)
		U.tofile(self.fp)
		V.tofile(self.fp)

		return True

	def close(self):
		self.fp.close()

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


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/kim/Desktop/ssd/HD_dataset")
parser.add_argument("--TTA", type=bool, default=False)
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints3/epoch_43_psnr_35.7481.pt") # network6 -> 34.44/0.9737
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints11/epoch_63_psnr_36.5063.pt") # network18 -> 35.28/0.9771
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt") # network19 -> 
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints15/epoch_98_psnr_36.0943.pt") # network22 -> , TTA: 32.593
# parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints24/epoch_280_psnr_36.4069.pt") # network37
parser.add_argument("--model_checkpoints", type=str, default="../finetune_model_checkpoints52/vimeo_epoch_270_psnr_36.3497.pt") # network55
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

model = Network()
# load_model_checkpoint(model, args.model_checkpoints)
load_model_checkpoint1(model, args.model_checkpoints)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")   

TTA = args.TTA
name_list = [
	(f'{args.path}/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
	(f'{args.path}/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
	(f'{args.path}/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
	(f'{args.path}/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
	(f'{args.path}/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
	(f'{args.path}/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
	(f'{args.path}/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
	(f'{args.path}/HD1080p_GT/BlueSky.yuv', 1080, 1920),
	(f'{args.path}/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
	(f'{args.path}/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
	(f'{args.path}/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),   
]

tot = []
for data in tqdm(name_list):
	psnr_list = []
	name = data[0]
	h, w = data[1], data[2]
	Reader = YUV_Read(os.path.join(args.path, name), h, w, toRGB=True)
	_, lastframe = Reader.read()

	for index in range(0, 100, 4):
		gt = []
		IMAGE1, success1 = Reader.read(index)
		IMAGE2, success2 = Reader.read(index + 4)
		if not success2:
			break
		# for i in range(1, 4):
		gt, _ = Reader.read(index + 2)
			# gt.append(tmp)

		I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
		I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
		
		padder = InputPadder(I0.shape, divisor=32)
		I0, I1 = padder.pad(I0, I1)
		# pred_list = model.multi_inference(I0, I1, TTA=TTA, time_list=[(i+1)*(1./4.) for i in range(3)], fast_TTA = TTA)
		
		# pred = model.inference(I0, I1)
		pred = model.forward(I0, I1)['I_t']
		pred = padder.unpad(pred)[0]
		
		# for i in range(len(pred_list)):
		#     pred_list[i] = padder.unpad(pred_list[i])

		# for i in range(3):
		out = (np.round(pred.detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
		diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
		mse = np.mean((diff_rgb - 128.0) ** 2)
		PIXEL_MAX = 255.0
		psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

		psnr_list.append(psnr)

	print(f'{data[0]}, {np.mean(psnr_list)}, total frame: {len(psnr_list)}')
	tot.append(np.mean(psnr_list))

print('PSNR: {}(544*1280), {}(720p), {}(1080p)'.format(np.mean(tot[0:4]), np.mean(tot[4:7]), np.mean(tot[7:])))