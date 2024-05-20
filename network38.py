import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import einops
from PIL import Image
from trainer import Trainer
from itertools import chain
from timm.models.layers import trunc_normal_
import math
import torchvision

import xformers
import xformers.ops

from flow_warp import flow_warp
from frameattn import MotionFormerBlock, Mlp

def upsample_flow(flow, upsample_factor=2, mode='bilinear'):
	if mode == 'nearest':
		up_flow = F.interpolate(flow, scale_factor=upsample_factor,
								mode=mode) * upsample_factor
	else:
		up_flow = F.interpolate(flow, scale_factor=upsample_factor,
								mode=mode, align_corners=True) * upsample_factor
	return up_flow	
	
def conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1):
	return nn.Sequential(
		nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride,
				  padding=padding, dilation=dilation, bias=True),
		nn.PReLU(out_dim)
		)

def deconv(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
	return nn.Sequential(
		nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, 
						   stride=stride, padding=padding, bias=True),
		nn.PReLU(out_dim)
		)    

class CrossScalePatchEmbed(nn.Module):
	def __init__(self, in_dims=[32, 64, 128, 256], fused_dim=None, conv=nn.Conv2d):
		super().__init__()
		
		layers = []
		for i in range(len(in_dims)-1):
			for j in range(2 ** i):
				layers.append(
					conv(in_channels=in_dims[-2-i],
						out_channels=in_dims[-2-i],
						kernel_size=3,
						stride=2**(i+1),
						padding=1+j,
						dilation=1+j,
						bias=True)
					)
		self.layers = nn.ModuleList(layers)
		concat_dim = sum([2**(2-i) * in_dims[i] for i in range(len(in_dims)-1)]) + in_dims[-1]
		if fused_dim is None:
			fused_dim = concat_dim
		self.proj = nn.Conv2d(concat_dim, fused_dim, 1, 1)
		self.norm = nn.LayerNorm(fused_dim)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, xs):
		ys = []
		k = 0
		for i in range(len(xs)-1):
			for _ in range(2 ** i):
				ys.append(self.layers[k](xs[-2-i]))
				k += 1
		ys.append(xs[-1])
		x = self.proj(torch.cat(ys, dim=1))
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)
		return x, H, W


class TransformerLayer(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop=0., mlp_ratio=4.):
		super().__init__()
		
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)

		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)	

	def forward(self, x):
		"""
		x: tensor with shape [B, H*W, C]
		return: tensor with shape [B, H*W, C]
		"""
		# MSA
		x_residual = self.norm1(x)
		qkv = self.qkv(x_residual)
		qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
		q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
		x_residual = xformers.ops.memory_efficient_attention(q, k, v)
		x_residual = einops.rearrange(x_residual, 'B L H D -> B L (H D)', H=self.num_heads)
		x_residual = self.proj(x_residual)
		x_residual = self.proj_drop(x_residual)

		x = x + x_residual
		x = x + self.mlp(self.norm2(x))
		return x


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()	
		self.pyramid_level = 4
		# self.hidden_dims = [16, 32, 64, 128]
		self.hidden_dims = [24, 48, 96, 192]
		# self.hidden_dims = [32, 64, 128, 256]
		assert len(self.hidden_dims) == self.pyramid_level
				
		# pyramid feature extraction
		self.feat_extracts = nn.ModuleList([])
		for i in range(self.pyramid_level):
			if i == 0:
				self.feat_extracts.append(
						nn.Sequential(conv(3, self.hidden_dims[i], kernel_size=3, stride=1, padding=1),
									  conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
					)
			else:
				self.feat_extracts.append(
						nn.Sequential(nn.AvgPool2d(kernel_size=2),
									  conv(self.hidden_dims[i-1], self.hidden_dims[i], kernel_size=3, stride=1, padding=1),
									  conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
					)
		
		concat_dim = sum([2**(2-i) * self.hidden_dims[i] for i in range(len(self.hidden_dims)-1)]) + self.hidden_dims[-1] # 320 or 480 or 640
		fused_dim = concat_dim
		self.cross_scale_feature_fusion = CrossScalePatchEmbed(in_dims=self.hidden_dims, fused_dim=fused_dim)
		
		window_size, num_heads, patch_size = 7, 8, 1
		self.translation_predictor = nn.ModuleList([
										MotionFormerBlock(dim=fused_dim, window_size=window_size, shift_size=0, 
														  patch_size=patch_size, num_heads=num_heads),
										MotionFormerBlock(dim=fused_dim, window_size=window_size, shift_size=window_size // 2, 
														  patch_size=patch_size, num_heads=num_heads)
									])
		
		self.fused_dim = fused_dim * 2
		self.motion_out_dim = 5
		motion_mlp_hidden_dim = int(self.fused_dim * 0.8) # 512 or 768 or 1024
		self.motion_mlp = nn.Sequential(
								conv(self.fused_dim + num_heads, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								conv(motion_mlp_hidden_dim, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								nn.Conv2d(motion_mlp_hidden_dim, self.motion_out_dim, kernel_size=1, stride=1, padding=0)
							)
				
		self.fused_dim1 = self.fused_dim // 2 # 320 or 480 or 640
		self.fused_dim2 = self.fused_dim // 4 # 160 or 240 or 320
		self.fused_dim3 = self.fused_dim // 8 # 80 or 120 or 160
		self.fused_dims = [self.fused_dim1, self.fused_dim2, self.fused_dim3, 2*self.fused_dim1]
		deconv_args = {'kernel_size':2, 'stride':2, 'padding':0}
		self.upsamples = nn.ModuleList([
							# HW: 32 -> 64
							nn.Sequential(deconv(self.fused_dim + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							# HW: 64 -> 128
							nn.Sequential(nn.PReLU(self.fused_dim1 + self.motion_out_dim),
										  deconv(self.fused_dim1 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							# HW: 128 -> 256
							nn.Sequential(nn.PReLU(self.fused_dim2 + self.motion_out_dim),
										  deconv(self.fused_dim2 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							])
		
		# residual refinement network
		in_chan = self.fused_dim3 + self.motion_out_dim + 15
		hidden_dim = 64
		# encoder
		self.proj = conv(in_chan, hidden_dim, kernel_size=3, stride=1, padding=1) # [256 -> 256]
		self.down1 = nn.Sequential( # [256 -> 128]
							conv(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
						)
		self.down2 = nn.Sequential( # [128 -> 64]
							# concat with backbone decoder's (128-size) output first
							conv(self.fused_dim2 + hidden_dim, 2 * hidden_dim, kernel_size=3, stride=2, padding=1),
							conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.down3 = nn.Sequential( # [64 -> 32]
							# concat with backbone decoder's (64-size) output first
							conv(self.fused_dim1 + 2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=2, padding=1),
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		# bottleneck
		self.bottleneck = nn.ModuleList([
								TransformerLayer(dim=4*hidden_dim, num_heads=8),
								TransformerLayer(dim=4*hidden_dim, num_heads=8)
							])
		# decoder
		self.up1 = nn.Sequential( # [32 -> 64]
							deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
							conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.up2 = nn.Sequential( # [64 -> 128]
							# concat with down2's output first
							deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
							conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.up3 = nn.Sequential( # [128 -> 256]
							# concat with down1's output first
							deconv(2 * hidden_dim, 1 * hidden_dim, kernel_size=2, stride=2, padding=0),
						)
		self.refine_head = nn.Sequential( # [256 -> 256]
							# concat with proj's output first
							conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
							conv(1 * hidden_dim, 3, kernel_size=3, stride=1, padding=1),
						)

	def visualize_feature(self, feat, scale_level, save_path="feat_visualize.png"):
		feat = feat.detach()
		feat = torch.mean(feat, dim=1)
		feat = feat[:, None, :, :]
		feat = (feat - feat.min()) / feat.max()

		feat = Trainer.convert_tensor_to_np(feat, [0.], [1.])
		feat = feat[8,:,:,0]
		
		feat_pil = Image.fromarray(feat)
		feat_pil = feat_pil.convert('L')
		feat_pil.save(f"scale_{scale_level}_{save_path}")

	def visualize_tensor(self, tensor, save_path="vis_tensor.png"):
		tensor = Trainer.convert_tensor_to_np(tensor)
		tensor = tensor[8]
		tensor_pil = Image.fromarray(tensor)
		tensor_pil = tensor_pil.convert('RGB')
		tensor_pil.save(f"{save_path}")


	def forward(self, im0, im1):
		'''
		im0, im1: tensor [B,3,H,W], float32, normalized to [0, 1]
		'''
		B,_,H,W = im0.size()
		im0_list = [im0]
		im1_list = [im1]
		im_t_list = []
		im0_warped_list = []
		im1_warped_list = []
		# downscale input frames
		for scale in range(self.pyramid_level-1):
			im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im0_list.append(im0_down)
			im1_list.append(im1_down)

		# CNN feature extraction
		feat_scale_level = []
		feat = torch.cat([im0, im1], dim=0) # speed up using parallelization
		for scale in range(self.pyramid_level):
			feat = self.feat_extracts[scale](feat)
			feat_scale_level.append(feat)

		# cross scale feature fusion
		feat, h, w = self.cross_scale_feature_fusion(feat_scale_level)
		feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)

		# window attention - acquire coarse motion and enhanced feature
		motion = []
		for k, blk in enumerate(self.translation_predictor):
			_, h, w, c = feat.size()
			feat, x_motion = blk(feat, h, w, B)
			if k == 0:
				feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)
			x_motion = einops.rearrange(x_motion, '(N B) L K -> B L (N K)', N=2)
			motion.append(x_motion)
		# [2*B, H*W, C] -> [B, 2*C, H, W]
		feat = einops.rearrange(feat, '(N B) (H W) C-> B (N C) H W', N=2, H=h)
		motion = torch.cat(motion, dim=2)
		motion = einops.rearrange(motion, 'B (H W) C -> B C H W', H=h)
		out = self.motion_mlp(torch.cat([motion, feat], dim=1))
		opt_flow_0 = out[:, :2]
		opt_flow_1 = out[:, 2:4]
		occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
		occ_mask2 = 1 - occ_mask1

		I_t_0 = flow_warp(im0_list[-1], opt_flow_0)
		I_t_1 = flow_warp(im1_list[-1], opt_flow_1)
		I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
		im0_warped_list.insert(0, I_t_0)
		im1_warped_list.insert(0, I_t_1)
		im_t_list.insert(0, I_t)
		
		# only warped feature once, 只warp一次、剩下的自己看著辦
		feat1 = flow_warp(feat[:, :self.fused_dims[0]], opt_flow_0)
		feat2 = flow_warp(feat[:, self.fused_dims[0]:self.fused_dims[-1]], opt_flow_1)
		feat = torch.cat([feat1, feat2, out], dim=1)

		backbone_decoder_feats = []

		# upscale motion along with feature
		for i, scale in enumerate(reversed(range(self.pyramid_level-1))):
			# forward features to get finer resolution
			feat = self.upsamples[i](feat) 
			out = feat[:, -self.motion_out_dim:] 
			opt_flow_0 = out[:, :2]
			opt_flow_1 = out[:, 2:4]
			occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
			occ_mask2 = 1 - occ_mask1

			if scale != 0:
				backbone_decoder_feats.append(feat[:, :-self.motion_out_dim] )

			I_t_0 = flow_warp(im0_list[scale], opt_flow_0)
			I_t_1 = flow_warp(im1_list[scale], opt_flow_1)
			I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
			im0_warped_list.insert(0, I_t_0)
			im1_warped_list.insert(0, I_t_1)
			im_t_list.insert(0, I_t)

		# residual refinement 
		# encoder
		feat0 = torch.cat([feat, im0, I_t_0, im1, I_t_1, I_t], dim=1) 
		feat0 = self.proj(feat0)
		feat1 = self.down1(feat0)  
		feat2 = self.down2(torch.cat([feat1, backbone_decoder_feats.pop()], dim=1)) 
		feat3 = self.down3(torch.cat([feat2, backbone_decoder_feats.pop()], dim=1)) 
		# bottleneck
		_, _, h, w = feat3.size()
		feat3 = einops.rearrange(feat3, 'B C H W -> B (H W) C') # patch embedding
		for blk in self.bottleneck:
			feat3 = blk(feat3)
		feat3 = einops.rearrange(feat3, 'B (H W) C -> B C H W', H=h) # patch unembedding
		# decoder
		feat2_ = self.up1(feat3) # 128
		feat1_ = self.up2(torch.cat([feat2_, feat2], dim=1))
		feat0_ = self.up3(torch.cat([feat1_, feat1], dim=1))
		# output
		I_t_residual = self.refine_head(torch.cat([feat0_, feat0], dim=1)) 
		I_t_residual = 2*torch.sigmoid(I_t_residual) - 1  # mapped to [-1, 1]
		I_t += I_t_residual
		I_t = torch.clamp(I_t, 0, 1)

		output_dict = {'I_t': I_t, 
				 	   'im_t_list': im_t_list, # scale: fine to coarse
					   'im0_warped_list': im0_warped_list,
					   'im1_warped_list': im1_warped_list,
					   'opt_flow_0': opt_flow_0,
					   'opt_flow_1': opt_flow_1,
					   'I_t_0': I_t_0,
					   'I_t_1': I_t_1,
					   'occ_mask1': occ_mask1,
					   'occ_mask2': occ_mask2,
					   }
		return output_dict
	
	@torch.no_grad()
	def inference(self, im0, im1):
		'''
		im0, im1: tensor [B,3,H,W], float32 normalized to [0, 1]
		'''
		B,_,_,_ = im0.size()

		# CNN feature extraction
		feat_scale_level = []
		feat = torch.cat([im0, im1], dim=0) # speed up using parallelization
		for scale in range(self.pyramid_level):
			feat = self.feat_extracts[scale](feat)
			feat_scale_level.append(feat)

		# cross scale feature fusion
		feat, h, w = self.cross_scale_feature_fusion(feat_scale_level)
		feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)

		# window attention - acquire coarse motion and enhanced feature
		motion = []
		k = 0
		for blk in self.translation_predictor:
			_, h, w, c = feat.size()
			feat, x_motion = blk(feat, h, w, B)
			if k == 0:
				feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)
				k += 1
			x_motion = einops.rearrange(x_motion, '(N B) L K -> B L (N K)', N=2)
			motion.append(x_motion)
		# [2*B, H*W, C] -> [B, 2*C, H, W]
		feat = einops.rearrange(feat, '(N B) (H W) C-> B (N C) H W', N=2, H=h)
		motion = torch.cat(motion, dim=2)
		motion = einops.rearrange(motion, 'B (H W) C -> B C H W', H=h)
		out = self.motion_mlp(torch.cat([motion, feat], dim=1))
		opt_flow_0 = out[:, :2]
		opt_flow_1 = out[:, 2:4]
		
		# only warped feature once, 只warp一次、剩下的自己看著辦
		feat1 = flow_warp(feat[:, :self.fused_dims[0]], opt_flow_0)
		feat2 = flow_warp(feat[:, self.fused_dims[0]:self.fused_dims[-1]], opt_flow_1)
		feat = torch.cat([feat1, feat2, out], dim=1)

		# upscale motion along with feature
		for i, scale in enumerate(reversed(range(self.pyramid_level-1))):
			# forward features to get finer resolution
			feat = self.upsamples[i](feat) 
		
		out = feat[:, -self.motion_out_dim:] 
		opt_flow_0 = out[:, :2]
		opt_flow_1 = out[:, 2:4]
		occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
		# occ_mask2 = torch.sigmoid(out[:, 5].unsqueeze(1))
		occ_mask2 = 1 - occ_mask1
		I_t_0 = flow_warp(im0, opt_flow_0)
		I_t_1 = flow_warp(im1, opt_flow_1)
		I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1

		# residual refinement 
		# encoder		
		feat0 = torch.cat([feat, im0, I_t_0, im1, I_t_1, I_t], dim=1) 
		feat1 = self.down1(feat0)  
		feat2 = self.down2(feat1) 
		feat3 = self.down3(feat2) 
		# decoder
		feat2_ = self.up1(feat3)
		feat2 = self.up1_fuse(torch.cat([feat2_, feat2], dim=1))
		feat1_ = self.up2(feat2)
		feat1 = self.up2_fuse(torch.cat([feat1_, feat1], dim=1))
		feat0_ = self.up3(feat1)
		feat0 = self.up3_fuse(torch.cat([feat0_, feat0], dim=1))
		# output
		I_t_residual = self.refine_out(feat0) 
		I_t_residual = 2*torch.sigmoid(I_t_residual) - 1  # mapped to [-1, 1]
		I_t += I_t_residual
		I_t = torch.clamp(I_t, 0, 1)

		return I_t
	
	
if __name__ == "__main__":
	from torchsummary import summary

	device = torch.device('cpu')
	# device = torch.device('cuda')

	unet = Network().to(device)
	pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M") 
	
	criterion = nn.MSELoss()
	optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4, weight_decay=1e-4)
	batchsize = 1

	a = torch.rand(batchsize,3,256,256).to(device)
	b = torch.rand(batchsize,3,256,256).to(device)
	c = torch.rand(batchsize,3,256,256).to(device)

	# summary(unet, [(3,256,256), (3,256,256)], device=device)

	for i in range(10):
		print(i)
		c_logits = unet(a,b)
		# print(c_logits.size())
		loss = criterion(c_logits, c)
		loss.backward()
		optimizer.step()
