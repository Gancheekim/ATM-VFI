
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pytorch_warmup as warmup
import flow_vis
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from loss_fn.vgg_loss import VGGPerceptualLoss
from loss_fn.laplacian_pyramid_loss import LapLoss
from loss_fn.census_loss import Ternary

class Trainer():
	def __init__(self, model, train_loader, val_loader, normalize_mean, normalize_std, args,
			  	 optim_checkpt=None):
		self.device             = torch.device(args.device)
		self.model              = model.to(self.device)
		self.train_loader       = train_loader
		self.val_loader         = val_loader
		self.num_epoch          = args.num_epoch
		self.model_checkpoints  = args.model_checkpoints
		self.visualization_path = args.visualization_path
		self.debug              = args.debug
		self.debug_iter         = args.debug_iter
		self.viz = args.viz
		self.normalize_mean     = normalize_mean
		self.normalize_std      = normalize_std
		self.visualize_idx      = 0

		# optimizer
		self.useGradientAccumulate   = False
		self.updateIter              = 2 if self.useGradientAccumulate else 1
		self.optimizer               = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
		if optim_checkpt is not None:
			self.optimizer.load_state_dict(optim_checkpt)
			if not args.resume_train:
				for g in self.optimizer.param_groups:
					g['lr'] = args.init_lr
				self.warmup_step = 500
				print(f'--- using saved optimizer state dict. (warmup step: {self.warmup_step}) ---')
			else:
				self.warmup_step = 50
				print(f'--- using saved optimizer state dict. (warmup step: {self.warmup_step}) ---')
		else:
			self.warmup_step = 2000
			print(f'--- NOT using saved optimizer state dict. (warmup step: {self.warmup_step}) ---')
		
		# gradient clipping
		self.isClipGradient   = False
		self.clip_max_norm    = 10.0
		
		# lr scheduler
		self.use_lr_scheduler   = True
		T_max = sum([len(data_loader) for data_loader in self.train_loader])
		T_max *= self.num_epoch // (len(self.train_loader) * self.updateIter)
		self.scheduler          = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=args.last_lr)	

		# warmup
		self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=self.warmup_step)	
		
		# activate loss function
		self.use_lap_loss           = True
		self.use_warping_loss       = True
		self.use_l1_loss            = False
		self.use_perceptual_loss    = True
		self.use_style_loss         = True
		# self.use_perceptual_loss    = False
		# self.use_style_loss         = False
		self.use_bidirect_warp_loss = False
		# loss weight
		self.lap_loss_weight           = 1.
		self.warping_loss_weight       = 0.25
		self.l1_loss_weight            = 1.
		self.perceptual_loss_weight    = 0.05
		self.style_loss_weight         = 5e-9
		self.bidirect_warp_loss_weight = 1.

		self.__setup_metric__()
		self.__setup_meta__(args)

	def __setup_metric__(self):
		self.train_metric = {"psnr": 0.}
		self.val_metric = {"psnr": 0.}
		if self.use_l1_loss:
			self.l1_loss = self.init_l1_loss(useCharbonnier=True)
			self.add_metric(self.train_metric, "l1_loss")
			self.add_metric(self.val_metric, "l1_loss")

		if self.use_lap_loss:
			self.lap_loss = LapLoss().to(self.device)
			self.add_metric(self.train_metric, "lap_loss")
			self.add_metric(self.val_metric, "lap_loss")

		if self.use_perceptual_loss or self.use_style_loss:
			self.perceptual_style_loss = self.init_perceptual_loss(self.use_perceptual_loss, self.use_style_loss)
			if self.use_perceptual_loss:
				self.add_metric(self.train_metric, "perceptual_loss")
				self.add_metric(self.val_metric, "perceptual_loss")
			if self.use_style_loss:
				self.add_metric(self.train_metric, "style_loss")
				self.add_metric(self.val_metric, "style_loss")
		if self.use_warping_loss:
			self.add_metric(self.train_metric, "warping_loss")
			self.add_metric(self.val_metric, "warping_loss")
		if self.use_bidirect_warp_loss:
			self.bidirect_warp_loss = Ternary(device=self.device)
			self.add_metric(self.train_metric, "bidirect_warp_loss")
			self.add_metric(self.val_metric, "bidirect_warp_loss")

		self.prev_train_metric = self.train_metric.copy()
		self.prev_val_metric = self.val_metric.copy()

	def __setup_meta__(self, args):
		self.meta = {"epoch": self.num_epoch,
					"batch_size": args.batch_size,
					"gradient accumulation": f'{self.updateIter} batch' if self.useGradientAccumulate else False,
					"optimizer": self.optimizer.__class__.__name__,
					"weight decay": args.weight_decay,
					"initial learning rate": args.init_lr,
					"final learning rate": args.last_lr,
					"learning rate scheduler": self.scheduler.__class__.__name__ if self.use_lr_scheduler else False,
					"warmup step": self.warmup_step,
					"use gradient clipping": f"True (max_norm: {self.clip_max_norm})" if self.isClipGradient else False,
					"train metric": self.train_metric,
					}
		for k,v in self.meta.items():
			print(f"{k}: {v}")


	@staticmethod
	def l1_with_charbonnier(pred, label, eps=1e-6):
		return torch.mean(torch.sqrt((pred - label).pow(2) + eps))
	
	def init_l1_loss(self, useCharbonnier=True):
		if not useCharbonnier:
			return nn.L1Loss()
		else:
			return self.l1_with_charbonnier
	
	def init_perceptual_loss(self, use_perceptual_loss, use_style_loss, vgg_type=19):
		return VGGPerceptualLoss(vgg_type=vgg_type, do_normalize=True,
						   		use_perceptual_loss=use_perceptual_loss, 
								use_style_loss=use_style_loss).to(self.device)
	
	def add_metric(self, metric_dict, metric_name):
		metric_dict[metric_name] = 0.
		return metric_dict
	
	def reset_metric(self, metric_dict):
		for k, _ in metric_dict.items():
			metric_dict[k] = 0.

	def normalize_metric(self, metric_dict, total_batch):
		for k, _ in metric_dict.items():
			metric_dict[k] /= total_batch
	
	def criterion(self, output, label, metric_dict):
		'''
		return L1 loss, VGG perceptual loss, Gram-matrix(Style) loss
		'''
		pred = output['I_t']
		loss_dict = {}
		loss = 0.
		if self.use_l1_loss:
			loss_dict['l1_loss'] = self.l1_loss_weight * self.l1_loss(pred, label)
			loss += loss_dict['l1_loss']
			metric_dict["l1_loss"] += loss_dict['l1_loss'].item()

		if self.use_lap_loss:
			loss_dict['lap_loss'] = self.lap_loss_weight * self.lap_loss(pred, label)
			loss += loss_dict['lap_loss']
			metric_dict["lap_loss"] += loss_dict['lap_loss'].item()

		if self.use_warping_loss:
			im_t_list = output['im_t_list']
			loss_dict['warping_loss'] = 0
			label_ = label.clone()
			for scale, im_t in enumerate(im_t_list):
				self.lap_loss.max_levels = min(5 - (scale-1), 5)
				# self.lap_loss.max_levels = min(5 - (scale-3), 5) # only for large motion
				loss_dict['warping_loss'] += self.lap_loss(im_t, label_)
				if scale < len(im_t_list) - 1:
					label_ = F.interpolate(label_, scale_factor=0.5, mode='bilinear', align_corners=True)
			loss_dict['warping_loss'] *= self.warping_loss_weight
			loss += loss_dict['warping_loss']
			metric_dict["warping_loss"] += loss_dict['warping_loss'].item()

		if self.use_perceptual_loss or self.use_style_loss:
			perceptual_loss, style_loss = self.perceptual_style_loss(pred.clone(), label.clone())
			loss_dict['perceptual_loss'] = self.perceptual_loss_weight * perceptual_loss
			loss_dict['style_loss'] = self.style_loss_weight * style_loss
			if self.use_perceptual_loss:
				loss += loss_dict['perceptual_loss']
				metric_dict["perceptual_loss"] += loss_dict['perceptual_loss'].item()
			if self.use_style_loss:
				loss += loss_dict['style_loss']
				metric_dict["style_loss"] += loss_dict['style_loss'].item()

		if self.use_pose_loss:
			loss_dict['pose_loss'] = self.pose_loss_weight * self.pose_loss(pred.clone(), label.clone())
			loss += loss_dict['pose_loss']
			metric_dict["pose_loss"] += loss_dict['pose_loss'].item()

		if self.use_sobel_loss:
			loss_dict['sobel_loss'] = self.sobel_loss_weight * self.sobel_loss(pred, label)
			loss += loss_dict['sobel_loss']
			metric_dict["sobel_loss"] += loss_dict['sobel_loss'].item()

		if self.use_bidirect_warp_loss:
			im0_list = output['im0_warped_list']
			im1_list = output['im1_warped_list']
			loss_dict['bidirect_warp_loss'] = 0.
			for i in range(len(im0_list)):
				loss_dict['bidirect_warp_loss'] += self.bidirect_warp_loss(im0_list[i], im1_list[i])
			loss_dict['bidirect_warp_loss'] *= self.bidirect_warp_loss_weight
			loss += loss_dict['bidirect_warp_loss']
			metric_dict["bidirect_warp_loss"] += loss_dict['bidirect_warp_loss'].item()

		return loss_dict, loss

	@staticmethod
	def cal_psnr(y_pred, y_true, MAX=1., reduction='mean'):
		"""
		args:
			y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
			y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
			MAX: the allowed max value of y_true/y_pred
		return: PSNR, larger the better
		"""
		mse = torch.mean((y_pred - y_true) ** 2 , [1,2,3]) # mse of each batch, thus size = [B]
		psnr = 10 * torch.log10(MAX / mse)
		if reduction == 'mean':
			psnr = torch.mean(psnr).item()
		else:
			psnr = psnr.tolist()
		return psnr

	def log_metrics_value(self):
		print("====== Training ======")
		log_msg = ""
		for i, (k, v) in enumerate(self.train_metric.items()):
			diff = round(self.train_metric[k] - self.prev_train_metric[k], 5)
			sign = '+' if diff > 0 else ''
			log_msg += f"{k}: {round(v, 5)}({sign}{diff})   "
			if i % 3 == 0:
				log_msg += '\n'
		print(log_msg)
		self.prev_train_metric = self.train_metric.copy()
		print("====== Validation ======")
		log_msg = ""
		for i, (k, v) in enumerate(self.val_metric.items()):
			diff = round(self.val_metric[k] - self.prev_val_metric[k], 5)
			sign = '+' if diff > 0 else ''
			log_msg += f"{k}: {round(v, 5)}({sign}{diff})   "
			if i % 3 == 0:
				log_msg += '\n'
		print(log_msg)
		self.prev_val_metric = self.val_metric.copy()

	def unnormalize_tensor(self, tensor, mean, std):
		'''
		args:
			tensor: tensor of shape [B, 3, H, W]
			mean/std: list of value (len==3)
		return:
			normalized tensor of shape [B, 3, H, W]
		'''
		tensor[:,0,:,:] =  std[0]*tensor[:,0,:,:] + mean[0]
		tensor[:,1,:,:] =  std[1]*tensor[:,1,:,:] + mean[1]
		tensor[:,2,:,:] =  std[2]*tensor[:,2,:,:] + mean[2]
		return tensor

	@staticmethod
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

	def save_prediction(self, im1, im3, im2_pred, im2_label, epoch, 
						opt_flow_0=None, opt_flow_1=None, psnr=[],
						I_t_0=None, I_t_1=None, occ_mask1=None, occ_mask2=None):
		'''
		args:
			- im1/im3/im2_pred/im2_label: tensor of shape [B, 3, H, W].
			- epoch: for naming the visualization result image.
			- psnr: list of len==batchsize, to record the psnr of im2_pred.
		'''
		mean=self.normalize_mean
		std=self.normalize_std
		im1 = self.convert_tensor_to_np(im1, mean, std)
		im3 = self.convert_tensor_to_np(im3, mean, std)
		im_overlay = (0.5*im1 + 0.5*im3).astype(np.uint8)
		im2_pred = self.convert_tensor_to_np(im2_pred, mean, std)
		im2_label = self.convert_tensor_to_np(im2_label, mean, std)
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
		I_t_0 = self.convert_tensor_to_np(I_t_0, mean, std)
		I_t_1 = self.convert_tensor_to_np(I_t_1, mean, std)
		occ_mask1 = self.convert_tensor_to_np(occ_mask1, [0.], [1.]).squeeze(3)
		occ_mask2 = self.convert_tensor_to_np(occ_mask2, [0.], [1.]).squeeze(3)

		B, _, _, C = im2_pred.shape		
		H, W = 256, 448
		for i in range(B):
			img = Image.new('RGB', (W*2,H*6))
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

			I_t_0_pil = Image.fromarray(I_t_0[i])
			I_t_1_pil = Image.fromarray(I_t_1[i])
			occ_mask1_pil = Image.fromarray(occ_mask1[i]).convert('L')
			occ_mask2_pil = Image.fromarray(occ_mask2[i]).convert('L')
			img.paste(I_t_0_pil.resize((W,H)), (0, 2*H))
			img.paste(I_t_1_pil.resize((W,H)), (W, 2*H))
			img.paste(occ_mask1_pil.resize((W,H)), (0, 3*H))
			img.paste(occ_mask2_pil.resize((W,H)), (W, 3*H))
			img.paste(im_overlay_pil.resize((W,H)), (W//2, 4*H))
			img.paste(im2_pred_pil.resize((W,H)), (0, 5*H))
			img.paste(im2_label_pil.resize((W,H)), (W, 5*H))
			if len(psnr) > 0:
				draw = ImageDraw.Draw(img)
				draw.text((20, int(4.8*H)), f"PSNR: {round(psnr[i], 3)}", font=ImageFont.truetype('FreeMono.ttf', 20), fill=(255,255,255))
			img.save(f"{self.visualization_path}epoch_{epoch}_idx_{self.visualize_idx}.png")
			self.visualize_idx += 1

	def update_optimizer_lrsched(self):
		self.optimizer.step()
		with self.warmup_scheduler.dampening():
			if self.use_lr_scheduler:
				self.scheduler.step()		

	def train(self):        
		for epoch in range(self.num_epoch):
			print(f'epoch = {epoch}, current LR: {"{:.3e}".format(self.scheduler.get_last_lr()[0])}')
			train_loader = self.train_loader[epoch % len(self.train_loader)]
			val_loader = self.val_loader[epoch % len(self.val_loader)]

			self.reset_metric(self.train_metric)
			self.model.train()
			for batch_idx, (im1, im2, im3) in enumerate(tqdm(train_loader)):
				if self.debug:
					if batch_idx == self.debug_iter:
						break

				im1 = im1.to(self.device)
				im2 = im2.to(self.device)
				im3 = im3.to(self.device)

				output = self.model(im1, im3)
				im2_pred = output['I_t']
				loss_dict, loss = self.criterion(output, im2, self.train_metric)

				# update model
				self.optimizer.zero_grad()
				loss.backward()
				if self.isClipGradient:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)

				if not self.useGradientAccumulate:
					self.update_optimizer_lrsched()
				elif ((batch_idx + 1) % self.updateIter == 0):
					self.update_optimizer_lrsched()

				# record evaluation metric (PSNR, SSIM, etc.)			
				self.train_metric["psnr"] += self.cal_psnr(im2_pred, im2)
			self.normalize_metric(self.train_metric, batch_idx+1)

			self.reset_metric(self.val_metric)
			self.model.eval()
			with torch.no_grad():
				for batch_idx, (im1, im2, im3) in enumerate(tqdm(val_loader)):
					if self.debug:
						if batch_idx == 5:
							break

					im1 = im1.to(self.device)
					im2 = im2.to(self.device)
					im3 = im3.to(self.device)

					output = self.model(im1, im3)
					im2_pred = output['I_t']
					loss_dict, loss = self.criterion(output, im2, self.val_metric)

					self.val_metric["psnr"] += self.cal_psnr(im2_pred, im2)

					if self.viz and (self.visualize_idx < 63):
					# save the inference result of the first batch for visualization
						im2_pred_ = im2_pred
						im2_ = im2
						psnr = self.cal_psnr(im2_pred, im2, reduction=None)
						opt_flow_0, opt_flow_1 = output['opt_flow_0'], output['opt_flow_1']
						self.save_prediction(im1, im3, im2_pred_, im2_, epoch, 
						   					 opt_flow_0, opt_flow_1, psnr,
											 output['I_t_0'], output['I_t_1'], 
											 output['occ_mask1'], output['occ_mask2'])
			self.visualize_idx = 0
			self.normalize_metric(self.val_metric, batch_idx+1)
			self.log_metrics_value()

			# save model checkpoint
			if epoch % len(self.train_loader) == 0:
				dataset_name = "vimeo_"
			elif epoch % len(self.train_loader) == 1:
				dataset_name = "x4k_snuEx_"
				
			torch.save(
				{
					'model_state_dict': self.model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict(),
					'meta_data': self.meta,
					'train_metric': self.train_metric,
					'val_metric': self.val_metric,
				},
				os.path.join(self.model_checkpoints, f'{dataset_name}epoch_{epoch}_psnr_{round(self.val_metric["psnr"],4)}.pt')
				)