import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

from finetune_trainer import Trainer
from dataset.vimeo_dataset import VimeoDataset
from dataset.X4K_dataset import X_Train, X_Test
from dataset.snu_dataset import SNUDataset

''' import model '''
# from network55 import Network
from network_base import Network
# from network_lite import Network

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
		if "relative_coord" in key:
			layers_to_remove.append(key)
		elif "attn_mask" in key:
			layers_to_remove.append(key)
		elif "HW" in key:
			layers_to_remove.append(key)
			
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param, strict=strict)
	return optim_checkpt


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--debug", action="store_true", default=False)
	parser.add_argument("--debug_iter", type=int, default=5) # iteration use for debug mode

	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--init_lr", type=float, default=4e-5)
	parser.add_argument("--last_lr", type=float, default=1e-5)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_epoch", type=int, default=300)
	parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument("--vimeo_path", type=str, default="/home/kim/ssd/vimeo_triplet/")
	parser.add_argument('--viz', action='store_true', default=False)
	parser.add_argument('--resume_train', action='store_true', default=False)
	parser.add_argument('--new_optimizer', action='store_true', default=False)

	parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints67/")
	parser.add_argument("--visualization_path", type=str, default="./finetune_visualization67/")

	args = parser.parse_args()

	myseed = 22112023
	torch.manual_seed(myseed)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
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
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")

	isLoadCheckpoint = True
	if isLoadCheckpoint:
		# network_base: to finetune on pretrained local and global part
		# param = "./finetune_model_checkpoints60/epoch_75_psnr_30.2463.pt" # pretrain global on X4k

		# network_base: to finetune from final checkpoint on perception quality
		param = "./finetune_model_checkpoints62/vimeo_epoch_254_psnr_36.3847.pt"
  
		# network_lite: to finetune on pretrained local and global part
		# param = "./finetune_model_checkpoints65/epoch_225_psnr_30.1287.pt"		

		optim_checkpt = load_model_checkpoint(model, param, strict=False)
		if args.new_optimizer:
			optim_checkpt = None
	else:
		optim_checkpt = None

	# --------------------------
	#		config model
	# --------------------------
	# phase 3: joint finetuning local and global part
	model.global_motion = True
	model.__finetune_local_motion__()
	model.__finetune_global_motion__()

	# --------------------------
	#		prepare data
	# --------------------------	
	train_loaders, val_loaders, batchsizes = [], [], []

	# Vimeo90K
	train_dataset = VimeoDataset(dataset_name='train', path="/home/kim/ssd/vimeo_triplet/")
	val_dataset = VimeoDataset(dataset_name='test', path="/home/kim/ssd/vimeo_triplet/")
	batchsizes.append(args.batch_size)
	train_loaders.append(train_dataset)
	val_loaders.append(val_dataset)

	# X4K
	train_dataset = X_Train(train_data_path="/home/kim/ssd/X4K1000FPS/train", min_t_step_size=2, max_t_step_size=32, random_crop=True, patch_size=448)
	val_dataset = SNUDataset(difficulty='hard') # can also be set to 'extreme'
	batchsizes.append(5)
	train_loaders.append(train_dataset)
	val_loaders.append(val_dataset)

	for i in range(len(train_loaders)):
		train_loaders[i] = DataLoader(train_loaders[i], batch_size=batchsizes[i], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
	for i in range(len(val_loaders)):
		val_loaders[i] = DataLoader(val_loaders[i], batch_size=batchsizes[i] if i == 0 else 1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

	# --------------------------
	#		config trainer
	# --------------------------
	trainer = Trainer(model=model, 
				   	  train_loader=train_loaders,
					  val_loader=val_loaders,
					  normalize_mean=[0., 0., 0.], 
					  normalize_std=[1., 1., 1.], 
					  args=args,
					  optim_checkpt=optim_checkpt)
	trainer.train() # start training