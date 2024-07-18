import cv2
import torch
import argparse
import warnings
import numpy as np
import sys
from benchmark.utils import InputPadder

''' import model '''
sys.path.append('./network/')
from network_base import Network as Network_base
from network_lite import Network as Network_lite

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
myseed = 22112023
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(myseed)  
	torch.cuda.manual_seed(myseed)

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
		if "attn_mask" in key:
			layers_to_remove.append(key)
		elif "HW" in key:
			layers_to_remove.append(key)
			
	for key in layers_to_remove:
		del param[key]
	model.load_state_dict(param, strict=strict)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M")  
	return optim_checkpt


def inference_2frame(img0, img1, model, isBGR=True):
	'''
	args:
		img0: first frame, numpy array, shape [H,W,3]
		img1: second frame, numpy array, shape [H,W,3]
		isBGR: channel order of input frame, bool

	return:
		pred: intermediate frame prediction, numpy array, shape [H,W,3]
	'''
	if isBGR:
		# BGR -> RBG
		img0 = img0[:, :, ::-1].copy()
		img1 = img1[:, :, ::-1].copy()

	# normalized from 0-255 to 0-1
	img0 = (torch.tensor(img0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
	img1 = (torch.tensor(img1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0) 
	
	# border padding
	padder = InputPadder(img0.shape, divisor=64)
	img0, img1 = padder.pad(img0, img1)

	# model forwarding
	pred = model.forward(img0, img1)["I_t"][0] 
	pred = padder.unpad(pred)
	pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
	pred = np.round(pred * 255).astype(np.uint8)

	if isBGR:
		# RGB -> BGR
		pred = pred[:,:,::-1].copy()

	return pred

def combine_frame_vert(upper, lower, FPS, 
					   text_location=(25, 35), 
					   font=cv2.FONT_HERSHEY_SIMPLEX, 
					   fontScale=1, 
					   fontColor=(255,255,255), 
					   thickness=1, 
					   lineType=2):
	upper_window = cv2.putText(upper.copy(), f'original {FPS} fps', text_location, font, fontScale, fontColor, thickness, lineType)
	lower_window = cv2.putText(lower.copy(), f'processed {2*FPS} fps', text_location, font, fontScale, fontColor, thickness, lineType)
	return np.concatenate((upper_window, lower_window), axis=0)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_type", type=str, default="base", choices=["base", "lite"])
	parser.add_argument("--ckpt", type=str, default="../research3_ckpt/atm-vfi-base-pct.pt") # atm-vfi-base.pt/atm-vfi-lite.pt/atm-vfi-base-pct.pt
	parser.add_argument("--global_off", action='store_true', default=False) # flag to turn off global motion estimation
	parser.add_argument("--video", type=str, default=None)
	parser.add_argument("--combine_video", action='store_true', default=False) # flag to horizontally concatenate input video and processed video
	parser.add_argument("--frame0", type=str, default=None)
	parser.add_argument("--frame1", type=str, default=None)
	parser.add_argument("--out", type=str, default="output_interpolation.png")
	args = parser.parse_args()

	if args.model_type == "base":
		model = Network_base()
	elif args.model_type == "lite":
		model = Network_lite()
	else:
		raise NotImplementedError

	load_model_checkpoint(model, args.ckpt)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device).eval()
	# enable/disable global motion estimation, default=True
	model.global_motion = not args.global_off

	# ================================
	#		video-based inference
	# ================================
	if args.video is not None:
		cap = cv2.VideoCapture(args.video)
		FPS = int(cap.get(cv2.CAP_PROP_FPS))
		W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		outVid = cv2.VideoWriter(args.out.replace('png', 'mp4'), fourcc, 2*FPS, (W,H))
		print(f'\noriginal video FPS: {FPS}, processed video FPS: {2*FPS}')

		if args.combine_video:
			outVidCombine = cv2.VideoWriter(args.out.replace('png', '') + '_combine.mp4', fourcc, 2*FPS, (W,2*H))

		frame_count = 0
		while cap.isOpened():
			print(f'processing frame: {frame_count}/{TOTAL_FRAME}', end='\r')
			read_success, curr_frame = cap.read()
			if read_success:
				if frame_count > 0:
					pred = inference_2frame(prev_frame, curr_frame, model, isBGR=True)
					outVid.write(prev_frame)
					outVid.write(pred)
					if args.combine_video:
						outVidCombine.write(combine_frame_vert(prev_frame.copy(), prev_frame.copy(), FPS))
						outVidCombine.write(combine_frame_vert(prev_frame.copy(), pred, FPS))
				else:
					isFirstFrame = False
				prev_frame = curr_frame.copy()
				frame_count += 1
				
			else:
				outVid.write(prev_frame)
				if args.combine_video:
					outVidCombine.write(combine_frame_vert(prev_frame.copy(), prev_frame.copy(), FPS))
				break
		
		cap.release()
		outVid.release()
		if args.combine_video:
			outVidCombine.release()

	# ================================
	#		frame-based inference
	# ================================
	elif args.frame0 is not None and args.frame1 is not None:
		img0 = cv2.imread(args.frame0)
		img1 = cv2.imread(args.frame1)
		pred = inference_2frame(img0, img1, model, isBGR=True)
		cv2.imwrite(args.out, pred)

	else:
		raise Exception("no input video or image argument are found. Use '--video', or '--frame0' and '--frame1' to specify the inputs.")