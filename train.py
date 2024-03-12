import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

from trainer import Trainer
from vimeo_dataset import VimeoDataset
from network37 import Network


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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	
	parser.add_argument("--debug", action="store_true", default=False)
	parser.add_argument("--debug_iter", type=int, default=5) # iteration use for debug mode

	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--init_lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_epoch", type=int, default=300)
	parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument("vimeo_path", type=str, default="/home/kim/ssd/vimeo_triplet/")

	parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints24/")
	parser.add_argument("--visualization_path", type=str, default="./finetune_visualization24/")

	args = parser.parse_args()

	myseed = 22112023
	torch.manual_seed(myseed)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	# torch.autograd.set_detect_anomaly(True)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(myseed)  
		torch.cuda.manual_seed(myseed)
		
	if not os.path.exists(args.model_checkpoints):
		os.makedirs(args.model_checkpoints)
	if not os.path.exists(args.visualization_path):
		os.makedirs(args.visualization_path)

	# --------------------------
	#		choose model
	# --------------------------	
	model = Network()
	# print(model)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")

	isLoadCheckpoint = False
	# isLoadCheckpoint = True
	if isLoadCheckpoint:
		param = "./finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt"
		load_model_checkpoint(model, param)

	# --------------------------
	#		prepare data
	# --------------------------	
	resize_shape = 256
	verticalFlip = False

	train_dataset = VimeoDataset(dataset_name='train', path="/home/kim/ssd/vimeo_triplet/")
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	val_dataset = VimeoDataset(dataset_name='test', path="/home/kim/ssd/vimeo_triplet/")
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

	# --------------------------
	#		config trainer
	# --------------------------
	trainer = Trainer(model, train_loader, val_loader, [0., 0., 0.], [1., 1., 1.], args)
	trainer.train() # start training