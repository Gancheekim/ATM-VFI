import re
import sys
import torch
import random
import numpy as np
from PIL import ImageFile
import torch.nn.functional as F
from imageio import imread, imwrite
import flow_vis
from PIL import Image, ImageDraw, ImageFont
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AverageMeter():
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class AverageMeterGroups:
	def __init__(self) -> None:
		self.meter_dict = dict()
	
	def update(self, dict, n=1):
		for name, val in dict.items():
			if self.meter_dict.get(name) is None:
				self.meter_dict[name] = AverageMeter()
			self.meter_dict[name].update(val, n)
	
	def reset(self, name=None):
		if name is None:
			for v in self.meter_dict.values():
				v.reset()
		else:
			meter = self.meter_dict.get(name)
			if meter is not None:
				meter.reset()
	
	def avg(self, name):
		meter = self.meter_dict.get(name)
		if meter is not None:
			return meter.avg


class InputPadder:
	""" Pads images such that dimensions are divisible by divisor """
	def __init__(self, dims, divisor=16):
		self.ht, self.wd = dims[-2:]
		pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
		pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
		self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

	def pad(self, *inputs):
		if len(inputs) == 1:
			return F.pad(inputs[0], self._pad, mode='replicate')
		else:
			return [F.pad(x, self._pad, mode='replicate') for x in inputs]

	def unpad(self, *inputs):
		if len(inputs) == 1:
			return self._unpad(inputs[0])
		else:
			return [self._unpad(x) for x in inputs]
	
	def _unpad(self, x):
		ht, wd = x.shape[-2:]
		c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
		return x[..., c[0]:c[1], c[2]:c[3]]


def img2tensor(img):
	if img.shape[-1] > 3:
		img = img[:,:,:3]
	return torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0


def tensor2img(img_t):
	return (img_t * 255.).detach(
						).squeeze(0).permute(1, 2, 0).cpu().numpy(
						).clip(0, 255).astype(np.uint8)

def seed_all(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def read(file):
	if file.endswith('.float3'): return readFloat(file)
	elif file.endswith('.flo'): return readFlow(file)
	elif file.endswith('.ppm'): return readImage(file)
	elif file.endswith('.pgm'): return readImage(file)
	elif file.endswith('.png'): return readImage(file)
	elif file.endswith('.jpg'): return readImage(file)
	elif file.endswith('.pfm'): return readPFM(file)[0]
	else: raise Exception('don\'t know how to read %s' % file)


def write(file, data):
	if file.endswith('.float3'): return writeFloat(file, data)
	elif file.endswith('.flo'): return writeFlow(file, data)
	elif file.endswith('.ppm'): return writeImage(file, data)
	elif file.endswith('.pgm'): return writeImage(file, data)
	elif file.endswith('.png'): return writeImage(file, data)
	elif file.endswith('.jpg'): return writeImage(file, data)
	elif file.endswith('.pfm'): return writePFM(file, data)
	else: raise Exception('don\'t know how to write %s' % file)


def readPFM(file):
	file = open(file, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip()
	if header.decode("ascii") == 'PF':
		color = True
	elif header.decode("ascii") == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
	if dim_match:
		width, height = list(map(int, dim_match.groups()))
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().decode("ascii").rstrip())
	if scale < 0:
		endian = '<'
		scale = -scale
	else:
		endian = '>'

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = np.reshape(data, shape)
	data = np.flipud(data)
	return data, scale


def writePFM(file, image, scale=1):
	file = open(file, 'wb')

	color = None

	if image.dtype.name != 'float32':
		raise Exception('Image dtype must be float32.')

	image = np.flipud(image)

	if len(image.shape) == 3 and image.shape[2] == 3:
		color = True
	elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
		color = False
	else:
		raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

	file.write('PF\n' if color else 'Pf\n'.encode())
	file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

	endian = image.dtype.byteorder

	if endian == '<' or endian == '=' and sys.byteorder == 'little':
		scale = -scale

	file.write('%f\n'.encode() % scale)

	image.tofile(file)


def readFlow(name):
	if name.endswith('.pfm') or name.endswith('.PFM'):
		return readPFM(name)[0][:,:,0:2]

	f = open(name, 'rb')

	header = f.read(4)
	if header.decode("utf-8") != 'PIEH':
		raise Exception('Flow file header does not contain PIEH')

	width = np.fromfile(f, np.int32, 1).squeeze()
	height = np.fromfile(f, np.int32, 1).squeeze()

	flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

	return flow.astype(np.float32)


def readImage(name):
	if name.endswith('.pfm') or name.endswith('.PFM'):
		data = readPFM(name)[0]
		if len(data.shape)==3:
			return data[:,:,0:3]
		else:
			return data
	return imread(name)


def writeImage(name, data):
	if name.endswith('.pfm') or name.endswith('.PFM'):
		return writePFM(name, data, 1)
	return imwrite(name, data)


def writeFlow(name, flow):
	f = open(name, 'wb')
	f.write('PIEH'.encode('utf-8'))
	np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
	flow = flow.astype(np.float32)
	flow.tofile(f)


def readFloat(name):
	f = open(name, 'rb')

	if(f.readline().decode("utf-8"))  != 'float\n':
		raise Exception('float file %s did not contain <float> keyword' % name)

	dim = int(f.readline())

	dims = []
	count = 1
	for i in range(0, dim):
		d = int(f.readline())
		dims.append(d)
		count *= d

	dims = list(reversed(dims))

	data = np.fromfile(f, np.float32, count).reshape(dims)
	if dim > 2:
		data = np.transpose(data, (2, 1, 0))
		data = np.transpose(data, (1, 0, 2))

	return data


def writeFloat(name, data):
	f = open(name, 'wb')

	dim=len(data.shape)
	if dim>3:
		raise Exception('bad float file dimension: %d' % dim)

	f.write(('float\n').encode('ascii'))
	f.write(('%d\n' % dim).encode('ascii'))

	if dim == 1:
		f.write(('%d\n' % data.shape[0]).encode('ascii'))
	else:
		f.write(('%d\n' % data.shape[1]).encode('ascii'))
		f.write(('%d\n' % data.shape[0]).encode('ascii'))
		for i in range(2, dim):
			f.write(('%d\n' % data.shape[i]).encode('ascii'))

	data = data.astype(np.float32)
	if dim==2:
		data.tofile(f)

	else:
		np.transpose(data, (2, 0, 1)).tofile(f)


def check_dim_and_resize(tensor_list):
	shape_list = []
	for t in tensor_list:
		shape_list.append(t.shape[2:])

	if len(set(shape_list)) > 1:
		desired_shape = shape_list[0]
		print(f'Inconsistent size of input video frames. All frames will be resized to {desired_shape}')
		
		resize_tensor_list = []
		for t in tensor_list:
			resize_tensor_list.append(torch.nn.functional.interpolate(t, size=tuple(desired_shape), mode='bilinear'))

		tensor_list = resize_tensor_list

	return tensor_list


def convert_tensor_to_np(tensor, mean=[0., 0., 0.], std=[1., 1., 1.]):
	'''
	args:
		tensor: tensor of shape [B, 3, H, W]
		mean/std: list of value (len==3)
	return:
		normalized numpy array of shape [B, H, W, 3], dtype=np.uint8
	'''
	np_arr = tensor.permute(0,2,3,1).cpu().numpy()
	for i in range(len(mean)):
		np_arr[:,:,:,i] =  std[i]*np_arr[:,:,:,i] + mean[i]
	np_arr *= 255
	np_arr = np.clip(np_arr, 0, 255)
	return np_arr.astype(np.uint8)

def save_prediction(im1, im3, im2_pred, im2_label, visualize_idx, visualization_path="./vimeo90k",
					opt_flow_0=None, opt_flow_1=None, psnr=[]):
	'''
	args:
		- im1/im3/im2_pred/im2_label: tensor of shape [B, 3, H, W].
		- psnr: list of len==batchsize, to record the psnr of im2_pred.
	'''
	if not os.path.exists(visualization_path):
		os.makedirs(visualization_path)

	mean = [0., 0., 0.]
	std = [1., 1., 1.]
	im1 = convert_tensor_to_np(im1, mean, std)
	im3 = convert_tensor_to_np(im3, mean, std)
	im_overlay = (0.5*im1 + 0.5*im3).astype(np.uint8)
	im2_pred = convert_tensor_to_np(im2_pred, mean, std)
	im2_label = convert_tensor_to_np(im2_label, mean, std)
	if (opt_flow_0 is not None) or (opt_flow_1 is not None):
		# visualize optical flow
		opt_flow_0 = opt_flow_0.detach().permute(0,2,3,1).cpu().numpy()
		opt_flow_1 = opt_flow_1.detach().permute(0,2,3,1).cpu().numpy()
		B,H,W,_ = opt_flow_0.shape
		opt_flow_0_rgb = np.zeros((B,H,W,3), dtype=np.uint8)
		opt_flow_1_rgb = np.zeros((B,H,W,3), dtype=np.uint8)
		for i in range(opt_flow_0.shape[0]):
			opt_flow_0_rgb[i] = flow_vis.flow_to_color(opt_flow_0[i], convert_to_bgr=False)
			opt_flow_1_rgb[i] = flow_vis.flow_to_color(opt_flow_1[i], convert_to_bgr=False)

	B, _, _, C = im2_pred.shape
	H, W = 256, 448
	for i in range(B):
		img = Image.new('RGB', (W*2,H*3))
		im1_pil = Image.fromarray(im1[i])
		im3_pil = Image.fromarray(im3[i])
		im_overlay_pil = Image.fromarray(im_overlay[i])
		im2_pred_pil = Image.fromarray(im2_pred[i])
		im2_label_pil = Image.fromarray(im2_label[i])
		img.paste(im1_pil.resize((W,H)), (0, 0))
		img.paste(im3_pil.resize((W,H)), (W, 0))
		if (opt_flow_0 is not None) or (opt_flow_1 is not None):
			opt_flow_0_pil = Image.fromarray(opt_flow_0_rgb[i])
			opt_flow_1_pil = Image.fromarray(opt_flow_1_rgb[i])
			img.paste(opt_flow_0_pil.resize((W,H)), (0, H))
			img.paste(opt_flow_1_pil.resize((W,H)), (W, H))
		
		img.paste(im_overlay_pil.resize((W,H)), (W//2, 1*H))
		img.paste(im2_pred_pil.resize((W,H)), (0, 2*H))
		img.paste(im2_label_pil.resize((W,H)), (W, 2*H))
		
		if len(psnr) > 0:
			draw = ImageDraw.Draw(img)
			draw.text((20, int(1.8*H)), f"PSNR: {round(psnr[i], 3)}", font=ImageFont.truetype('FreeMono.ttf', 20), fill=(255,255,255))
		img.save(f"{visualization_path}/idx_{visualize_idx}.png")
