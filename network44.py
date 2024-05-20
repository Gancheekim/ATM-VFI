import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops
import math
from flow_warp import flow_warp
from frameattn import MotionFormerBlock, RefineBottleneck

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
		
		self.extra_encoder_block = nn.Sequential(
									conv(fused_dim, fused_dim, kernel_size=3, stride=2, padding=1),
									conv(fused_dim, fused_dim, kernel_size=3, stride=1, padding=1)
									)
		
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

		self.extra_decoder_block = nn.Sequential(deconv(self.fused_dim + self.motion_out_dim, self.fused_dim + self.motion_out_dim, 
												 		kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
												conv(self.fused_dim + self.motion_out_dim, self.fused_dim + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
												nn.Conv2d(self.fused_dim + self.motion_out_dim, self.fused_dim + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  )
		
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
		self.down4 = nn.Sequential( # [32 -> 16]
							# concat with backbone decoder's (32-size) output first
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=2, padding=1),
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		# bottleneck
		window_size, num_heads, patch_size = 8, 8, 1
		self.bottleneck = nn.ModuleList([
								RefineBottleneck(dim=4*hidden_dim, window_size=window_size, shift_size=0, 
													patch_size=patch_size, num_heads=num_heads),
								RefineBottleneck(dim=4*hidden_dim, window_size=window_size, shift_size=window_size // 2, 
													patch_size=patch_size, num_heads=num_heads)
							])
		# decoder
		self.up0 = nn.Sequential( # [16 -> 32]
							deconv(4 * hidden_dim, 4 * hidden_dim, kernel_size=2, stride=2, padding=0),
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
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

		im0_smallest = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
		im1_smallest = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)

		# CNN feature extraction
		feat_scale_level = []
		feat = torch.cat([im0, im1], dim=0) # speed up using parallelization
		for scale in range(self.pyramid_level):
			feat = self.feat_extracts[scale](feat)
			feat_scale_level.append(feat)

		# cross scale feature fusion
		feat, h, w = self.cross_scale_feature_fusion(feat_scale_level)
		feat = einops.rearrange(feat, 'B (H W) C -> B C H W', H=h)
		feat = self.extra_encoder_block(feat)
		feat = einops.rearrange(feat, 'B C H W -> B H W C', H=h//2)

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

		I_t_0 = flow_warp(im0_smallest, opt_flow_0)
		I_t_1 = flow_warp(im1_smallest, opt_flow_1)
		I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
		im0_warped_list.insert(0, I_t_0)
		im1_warped_list.insert(0, I_t_1)
		im_t_list.insert(0, I_t)
		
		# only warped feature once, 只warp一次、剩下的自己看著辦
		feat1 = flow_warp(feat[:, :self.fused_dims[0]], opt_flow_0)
		feat2 = flow_warp(feat[:, self.fused_dims[0]:self.fused_dims[-1]], opt_flow_1)
		feat = torch.cat([feat1, feat2, out], dim=1)

		feat = self.extra_decoder_block(feat) 
		out = feat[:, -self.motion_out_dim:] 
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
		feat3 = self.down4(feat3) 
		# bottleneck
		feat3 = einops.rearrange(feat3, 'B C H W -> B H W C')
		_, h, w, _ = feat3.size()
		for k, blk in enumerate(self.bottleneck):
			feat3 = blk(feat3)
			if k == 0:
				feat3 = einops.rearrange(feat3, 'B (H W) C -> B H W C', H=h)
			else:
				feat3 = einops.rearrange(feat3, 'B (H W) C -> B C H W', H=h)
		# decoder
		feat3 = self.up0(feat3)
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

		# residual refinement 
		# encoder
		feat0 = torch.cat([feat, im0, I_t_0, im1, I_t_1, I_t], dim=1) 
		feat0 = self.proj(feat0)
		feat1 = self.down1(feat0)  
		feat2 = self.down2(torch.cat([feat1, backbone_decoder_feats.pop()], dim=1)) 
		feat3 = self.down3(torch.cat([feat2, backbone_decoder_feats.pop()], dim=1)) 
		# bottleneck
		feat3 = einops.rearrange(feat3, 'B C H W -> B H W C')
		_, h, w, _ = feat3.size()
		for k, blk in enumerate(self.bottleneck):
			feat3 = blk(feat3)
			if k == 0:
				feat3 = einops.rearrange(feat3, 'B (H W) C -> B H W C', H=h)
			else:
				feat3 = einops.rearrange(feat3, 'B (H W) C -> B C H W', H=h)
		# decoder
		feat2_ = self.up1(feat3) # 128
		feat1_ = self.up2(torch.cat([feat2_, feat2], dim=1))
		feat0_ = self.up3(torch.cat([feat1_, feat1], dim=1))
		# output
		I_t_residual = self.refine_head(torch.cat([feat0_, feat0], dim=1)) 
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
