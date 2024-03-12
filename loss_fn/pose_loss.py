import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ultralytics import YOLO
from easy_ViTPose.vit_models.model import ViTPose
from easy_ViTPose.vit_utils.util import dyn_model_import, infer_dataset_by_path

'''
code borrowed from:
- easy_ViTPose (https://github.com/JunkyByte/easy_ViTPose)
- ultralytics (https://github.com/ultralytics/ultralytics)
'''

class PoseLoss(torch.nn.Module):
	def __init__(self, 
				 yolo_path="/home/kim/Desktop/ssd/research3/saved_checkpoint/yolov8s.pt", 
				 model_path="/home/kim/Desktop/ssd/research3/saved_checkpoint/vitpose-b-coco.pth", 
				 model_sz="b", 
				#  loss=nn.CrossEntropyLoss(),
				 max_batch_size=64):
		super(PoseLoss, self).__init__()

		self.loss = nn.CrossEntropyLoss(reduction='none')
		self.max_batch_size = max_batch_size
		self.register_buffer("pad_bbox", torch.tensor([-10, 10]))
		self.register_buffer("MEAN", torch.tensor([0.485, 0.456, 0.406]))
		self.register_buffer("STD", torch.tensor([0.229, 0.224, 0.225]))
		self.resize_H = 256
		self.resize_W = 192
		# YOLO detection
		self.yolo_model = YOLO(model=yolo_path, task='detect')
		self.yolo_img_sz = 320
		# ViT pose estimation for 17 human keypoint
		dataset = infer_dataset_by_path(model_path)
		model_cfg = dyn_model_import(dataset, model=model_sz)
		self.vit_pose = ViTPose(model_cfg)
		vit_pose_ckpt = torch.load(model_path)
		if 'state_dict' in vit_pose_ckpt:
			self.vit_pose.load_state_dict(vit_pose_ckpt['state_dict'])
		else:
			self.vit_pose.load_state_dict(vit_pose_ckpt)
		# freeze both model
		for param in self.yolo_model.parameters():
			param.requires_grad = False
		for param in self.vit_pose.parameters():
			param.requires_grad = False

	def pad_image(self, image, aspect_ratio):
		assert len(image.size()) == 4, "input tensor must have shape [B, 3, H, W]"
		_, _, image_height, image_width = image.size()
		current_aspect_ratio = image_width / image_height

		left_pad = 0
		top_pad = 0
		# Determine whether to pad horizontally or vertically
		if current_aspect_ratio < aspect_ratio:
			# Pad horizontally
			target_width = int(aspect_ratio * image_height)
			pad_width = target_width - image_width
			left_pad = pad_width // 2
			right_pad = pad_width - left_pad

			padded_image = F.pad(image,
								pad=(left_pad, right_pad, 0,0, 0, 0, 0, 0),
								mode='constant')
		else:
			# Pad vertically
			target_height = int(image_width / aspect_ratio)
			pad_height = target_height - image_height
			top_pad = pad_height // 2
			bottom_pad = pad_height - top_pad

			padded_image = F.pad(image,
								pad=(0,0, top_pad, bottom_pad, 0, 0, 0, 0),
								mode='constant')
		return padded_image, (left_pad, top_pad)

	def detect_box(self, img):
		'''
		img: tensor shape [B, 3, H, W], value range [0, 1]
		
		return: list of <ultralytics.engine.results.Results>, length == B
		'''
		img = torch.clamp(img, 0., 1.)
		return self.yolo_model(img, imgsz=self.yolo_img_sz, verbose=False)
	
	def process_bboxes(self, bboxes, H, W, conf_thresh=0.35):
		'''
		bboxes: list of <ultralytics.engine.results.Results>, length == B

		return: list of tensor with shape [N, 6], length of list == B
				- N: how many filtered detected bboxes
				- 6: [xmin, ymix, xmax, ymax, confidence, class]
		'''
		processes_bboxes = []
		for result in bboxes:
			# traverse and filter bounding box
			result = result.boxes.data
			if result.size()[0] > 0:
				index = [i for i in range(result.size()[0]) if (result[i,4] >= conf_thresh) and (result[i,5] == 0)]
				result = torch.round(result[index]).long()
				result[:, [0,2]] = torch.clamp(result[:, [0,2]] + self.pad_bbox, 0, W)
				result[:, [1,3]] = torch.clamp(result[:, [1,3]] + self.pad_bbox, 0, H)
				processes_bboxes.append([result])
			else:
				processes_bboxes.append([])
		return processes_bboxes
	
	def get_cropped_img(self, processed_bboxes, img):
		for i in range(3):
			img[:,i,:,:] = (img[:,i,:,:] - self.MEAN[i]) / self.STD[i]

		preprocessed_img = []
		for batch, bboxes in enumerate(processed_bboxes):
			if len(bboxes) == 0:
				continue
			bboxes = bboxes[0]
			for bbox in bboxes:
				img_crop = img[batch, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
				img_crop, (left_pad, top_pad) = self.pad_image(img_crop[None], 3 / 4)
				# preprocess: crop to specific size 
				img_crop = F.interpolate(img_crop, size=(self.resize_H, self.resize_W), mode='bilinear', align_corners=True)
				preprocessed_img.append(img_crop)
		preprocessed_img = preprocessed_img[:self.max_batch_size]
		if len(preprocessed_img) == 0:
			# need to handle situation where no human is detected
			return []
		preprocessed_img = torch.cat(preprocessed_img, dim=0)
		return preprocessed_img

	def get_heatmap(self, cropped_img):
		'''
		cropped_img: tensor shape [B, 3, resize_H, resize_W]
		
		return: tensor shape [B, 17, resize_H//4, resize_W//4]
		'''
		return self.vit_pose(cropped_img)
	
	def forward(self, img_synthesis, img_gt, mode=2):
		if mode == 1:
			return self.forward1(img_synthesis, img_gt)
		elif mode == 2:
			return self.forward2(img_synthesis, img_gt)


	def forward1(self, img_synthesis, img_gt):
		'''
		cross-entropy loss with masking,
		mask is created by thresholding: 
			- valid keypoint label (max logit > threshold)
			- position with logits > threshold * max_logit_of_the_label
		'''
		B,C,H,W = img_gt.size()
		with torch.no_grad():
			img_gt = img_gt.detach()
			bboxes = self.detect_box(img_gt)
			processed_bboxes = self.process_bboxes(bboxes, H=H, W=W)
			cropped_img_gt = self.get_cropped_img(processed_bboxes, img_gt)
			if len(cropped_img_gt) == 0:
				# handling no detected human
				return torch.tensor(0.).to(img_gt.device)
			heatmaps_gt = self.get_heatmap(cropped_img_gt)
			keypt_label_gt = heatmaps_gt.argmax(dim=1)
			mask = self.get_mask(heatmaps_gt)

		cropped_img = self.get_cropped_img(processed_bboxes, img_synthesis)
		heatmaps = self.get_heatmap(cropped_img)

		loss = self.loss(heatmaps, keypt_label_gt)
		loss = torch.mean(loss * mask)
		return loss

	def forward2(self, img_synthesis, img_gt):
		'''
		no masking, but using KL-Divergence
		'''
		B,C,H,W = img_gt.size()
		img_gt = img_gt.detach()
		bboxes = self.detect_box(img_gt)
		processed_bboxes = self.process_bboxes(bboxes, H=H, W=W)
		cropped_img_gt = self.get_cropped_img(processed_bboxes, img_gt)
		if len(cropped_img_gt) == 0:
			# handling case with no detected human
			return torch.tensor(0.).to(img_gt.device)
		heatmaps_gt = self.get_heatmap(cropped_img_gt)

		cropped_img = self.get_cropped_img(processed_bboxes, img_synthesis)
		heatmaps = self.get_heatmap(cropped_img)

		# KL-divergence
		pred = F.log_softmax(heatmaps, dim=1)
		target = F.softmax(heatmaps_gt, dim=1)
		loss = F.kl_div(pred, target, log_target=False)
		return loss    

	def get_mask(self, heatmaps, threshold=0.9, kp_threshold=1.2):
		B,_,H,W = heatmaps.size()
		mask = torch.zeros(B,1,H,W).to(heatmaps.device)
		for i in range(B):
			heatmap = heatmaps[None, i].clone()
			test = heatmap.clone()
			a, _ = torch.max(test.view(1,17,-1), dim=2) # a: 17 class's max prob.
			
			heatmap, cls_logit = torch.max(heatmap, dim=1)

			valid_kp = []
			invalid_kp = []
			for j in range(17):
				if a[0, j] > kp_threshold:
					valid_kp.append(j)
				else:
					invalid_kp.append(j)

			for label in valid_kp:
				heatmap[cls_logit == label] = torch.where(heatmap[cls_logit == label] < threshold * a[0, label], 0., 1.0)
			for label in invalid_kp:
				heatmap[cls_logit == label] = 0.
			mask[i, :] = heatmap
		return mask.squeeze(1)

	def get_heatmap_project_back(self, img, threshold1=1.0, threshold2=0.9):
		B,C,H,W = img.size()
		bboxes = self.detect_box(img)
		processed_bboxes = self.process_bboxes(bboxes, H=H, W=W) # [B, N, 6]

		cropped_img = self.get_cropped_img(processed_bboxes, img) 
		heatmaps = self.get_heatmap(cropped_img) # [sum(N_i), 17, H', W']

		result = torch.zeros(B, 1, H, W)
		tmp = torch.zeros(B, 1, H, W)
		idx = 0
		for batch, bboxes in enumerate(processed_bboxes):
			if len(bboxes) == 0:
				continue
			bboxes = bboxes[0]
			for bbox in bboxes:
				h = bbox[3] - bbox[1]
				w = bbox[2] - bbox[0]
				heatmap = heatmaps[None, idx]
				# heatmap = F.softmax(heatmap, dim=1)
				# heatmap = (heatmap - torch.min(heatmap)) / torch.max(heatmap)
				heatmap = F.interpolate(heatmap, size=(h,w), mode='bilinear', align_corners=True)

				test = heatmap.clone()
				a, b = torch.max(test.view(1,17,-1), dim=2)
				# a: 17 class's max prob.
				
				heatmap, cls_logit = torch.max(heatmap, dim=1)
				
				peak = torch.max(heatmap[0])
				null = torch.min(heatmap[0])
				for i in range(h):
					for j in range(w):
						if heatmap[0, i, j] < 0.9 * a[0, cls_logit[0, i, j]]:
							heatmap[0, i, j] = null 

				# valid_kp = []
				# invalid_kp = []
				# for i in range(17):
				#     if a[0, i] > threshold1:
				#         valid_kp.append(i)
				#     else:
				#         invalid_kp.append(i)

				# for label in valid_kp:
				#     heatmap[cls_logit == label] = torch.where(heatmap[cls_logit == label] < threshold2 * a[0, label], 0., 1.0)
				# for label in invalid_kp:
				#     heatmap[cls_logit == label] = 0.
				
				# tmp[batch, 0, :, :] = 0
				# tmp[batch, 0, bbox[1]:bbox[3], bbox[0]:bbox[2]] = heatmap

				# prev = result[batch, 0, bbox[1]:bbox[3], bbox[0]:bbox[2]]

				# print(heatmap.size())

				for x, i in enumerate(range(bbox[1],bbox[3])):
					for y, j in enumerate(range(bbox[0],bbox[2])):
						if heatmap[0, x, y] > result[batch, 0, i, j]:
							result[batch, 0, i, j] = heatmap[0, x, y]

						# else:
							# result[batch, 0, i, j] = heatmap[0, x, y]
						# result[batch, 0, i, j] = max(result[batch, 0, i, j], heatmap[0, x, y])
				

				# result[batch, 0, bbox[1]:bbox[3], bbox[0]:bbox[2]] = heatmap
				idx += 1

		return result
	

if __name__ == "__main__":
	vimeo_path = "/home/kim/Desktop/ssd/vimeo_triplet/sequences/"
	# select_path = "00052/0016/"
	# select_path = "00001/0581/"
	select_path = "00001/0778/"

	img1 = cv2.imread(f"{vimeo_path+select_path}im1.png")

	cv2.imwrite("nba1.png", img1)
	# img1 = cv2.imread("nba_distort.png")
	# img1 = cv2.resize(img1, (448, 256))

	# img_32x32 = cv2.resize(img1.copy(), (32, 32))
	# cv2.imwrite("nba_32x32.png", img_32x32)

	# img_64x64 = cv2.resize(img1.copy(), (64, 64))
	# cv2.imwrite("nba_64x64.png", img_64x64)

	img_128x128 = cv2.resize(img1.copy(), (128, 128))
	cv2.imwrite("nba1_128x128.png", img_128x128)


	img3 = cv2.imread(f"{vimeo_path+select_path}im3.png")
	img = np.stack([img1, img3], axis=0)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cpu')

	img_gt = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.
	img_gt = img_gt.to(device)

	B,C,H,W = img_gt.size()
	img_tensor = torch.rand(B,C,H,W).float().to(device)

	yolo_path = './saved_checkpoint/yolov8s.pt'
	model_path = './saved_checkpoint/vitpose-b-coco.pth'
	pose_loss = PoseLoss(yolo_path, model_path, model_sz='b').to(device)

	# for param in pose_loss.yolo_model.parameters():
	# 	param.requires_grad = True
	# for param in pose_loss.vit_pose.parameters():
	# 	param.requires_grad = True
	# pytorch_total_params = sum(p.numel() for p in pose_loss.yolo_model.parameters() if p.requires_grad)
	# print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")
	# pytorch_total_params = sum(p.numel() for p in pose_loss.vit_pose.parameters() if p.requires_grad)
	# print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")

	# loss = pose_loss(img_tensor, img_gt)
	# print(loss)

	# heatmaps = pose_loss.get_heatmap_project_back(img_gt, threshold1=1.0, threshold2=0.9)
	heatmaps = pose_loss.get_heatmap_project_back(img_gt, threshold1=0.1, threshold2=0.1)

	heatmaps_np = heatmaps.detach().clone().cpu().numpy()[0,0]
	print(np.min(heatmaps_np))
	print(np.max(heatmaps_np))
	heatmaps_np = (heatmaps_np - np.min(heatmaps_np)) / np.max(heatmaps_np)

	plt.imshow(heatmaps_np, cmap='hot', interpolation='nearest')
	# plt.show()
	plt.savefig("heatmap.png")

	# # heatmaps_np = cv2.cvtColor(heatmaps_np, cv2.COLOR_GRAY2RGB)
	# # print(heatmaps_np.shape)

	# colormap = plt.get_cmap('inferno')
	# heatmaps_np = (colormap(heatmaps_np) * 2**16).astype(np.uint16)[:,:,:3]
	# # heatmaps_np = cv2.cvtColor(heatmaps_np, cv2.COLOR_RGB2BGR)

	# print(heatmaps_np.shape)

	# # cv2.imwrite("heatmap.png", heatmaps_np)


	# from PIL import Image
	# heatmap_pil = Image.fromarray(heatmaps_np).convert('RGB')
	# heatmap_pil.resize((448, 256))
	# heatmap_pil.save("heatmap.png")