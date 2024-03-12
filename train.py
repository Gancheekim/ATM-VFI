import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

from trainer import Trainer
# from data_preprocess import Prepare_dataset
from vimeo_dataset import VimeoDataset
# from network import Network
# from network1 import Network
# from network2 import Network
# from network3 import Network
# from network4 import Network
# from network5 import Network
# from network6 import Network
# from network7 import Network # second best
# from network8 import Network
# from network9 import Network
# from network10 import Network
# from network11 import Network
# from network12 import Network
# from network13 import Network
# from network14 import Network
# from network15 import Network
# from network17 import Network
# from network18 import Network
# from network19 import Network
# from network20 import Network
# from network21 import Network
# from network23 import Network
# from network22 import Network
# from network24 import Network
# from network25 import Network
# from network26 import Network
# from network27 import Network
# from network28 import Network
# from network30 import Network
# from network31 import Network
# from network33 import Network
# from network34 import Network
# from network35 import Network
# from network36 import Network
from network37 import Network

# from film.models import film_net, FILMNet_Weights
# from film_models import film_net, FILMNet_Weights


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

	parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument("--debug", action="store_true", default=False) # 64  (RIFE)
	parser.add_argument("--debug_iter", type=int, default=5)

	parser.add_argument("--batch_size", type=int, default=16) # 16 # 64  (RIFE)
	parser.add_argument("--init_lr", type=float, default=1e-4) # 1e-4 to 1e-5 (RIFE) @ 2e-4
	parser.add_argument("--weight_decay", type=float, default=1e-4) # 1e-4 (RIFE)
	parser.add_argument("--num_epoch", type=int, default=300)

	# parser.add_argument("--model_checkpoints", type=str, default="./model_checkpoints/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints2/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints3/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints4/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints5/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints6/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints7/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints9/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints10/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints11/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints12/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints13/")
	# parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints17/")
	parser.add_argument("--model_checkpoints", type=str, default="./finetune_model_checkpoints24/")
	
	# parser.add_argument("--visualization_path", type=str, default="./visualization/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization2/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization3/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization4/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization5/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization6/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization7/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization9/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization10/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization11/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization12/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization13/")
	# parser.add_argument("--visualization_path", type=str, default="./finetune_visualization17/")
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
		# model = Network(train_flow_loss=True, train_warping_loss=True)
		# param = torch.load("./model_checkpoints/epoch_9_loss_29.6028.pt", map_location='cuda:0')
		# model.load_state_dict(param)
		# '''
		# param = torch.load("./model_checkpoints/epoch_38_loss_33.9185.pt", map_location='cuda:0')
		# param = torch.load("./finetune_model_checkpoints4/epoch_0_psnr_30.7595.pt", map_location='cuda:0')
		# param = torch.load("./finetune_model_checkpoints5/epoch_1_psnr_34.3393.pt", map_location='cuda:0')
		# param = torch.load("./finetune_model_checkpoints5/epoch_99_psnr_35.5483.pt", map_location='cuda:0')
		# param = torch.load("./finetune_model_checkpoints8/epoch_4_psnr_35.0955.pt", map_location='cuda:0')
		# param = torch.load("./finetune_model_checkpoints9/epoch_6_psnr_34.3777.pt", map_location='cuda:0')
		param = "./finetune_model_checkpoints12/epoch_62_psnr_35.9664.pt"
		
		'''
		# remove layer
		layers_to_remove = []
		for key in param:
			if "translation_predictor" in key:
				layers_to_remove.append(key)
		for key in layers_to_remove:
			del param[key]
		'''
		# model.load_state_dict(param)
		load_model_checkpoint(model, param)
		# model.pretrained_flow_net = nn.Identity()
		# '''

	# model = Network(train_flow=False)
	# print(model)

	# model = film_net(weights=FILMNet_Weights.DEFAULT)
	# model.to(args.device)
	# model.eval()

	# precision = torch.float32
	# model_path = "./frame-interpolation-pytorch/film_net_fp32.pt"
	# device = torch.device('cuda')
	# model = torch.jit.load(model_path, map_location='cpu')
	# model.to(device=device, dtype=precision)

	# --------------------------
	#		prepare data
	# --------------------------	
	resize_shape = 256
	verticalFlip = False

	# train_dataset = Prepare_dataset(mode='train', data_list_txt="/home/kim/ssd/vimeo_triplet/tri_trainlist.txt", resize_shape=resize_shape, verticalFlip=verticalFlip)
	train_dataset = VimeoDataset(dataset_name='train', path="/home/kim/ssd/vimeo_triplet/")
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	# val_dataset = Prepare_dataset(mode='val', data_list_txt="/home/kim/ssd/vimeo_triplet/tri_testlist.txt", resize_shape=resize_shape)
	# val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=4)
	val_dataset = VimeoDataset(dataset_name='test', path="/home/kim/ssd/vimeo_triplet/")
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

	# --------------------------
	#		config trainer
	# --------------------------
	# trainer = Trainer(model, train_loader, val_loader, val_dataset.normalize_mean, val_dataset.normalize_std, args)
	trainer = Trainer(model, train_loader, val_loader, [0., 0., 0.], [1., 1., 1.], args)
	trainer.train()