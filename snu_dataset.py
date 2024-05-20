import os
import cv2
import torch
from torch.utils.data import Dataset
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


class SNUDataset(Dataset):
	def __init__(self, 
			  	difficulty='hard',
			  	path="/home/kim/Desktop/ssd/snufilm-test/eval_modes/",
			  	img_data_path="/home/kim/Desktop/ssd/snufilm-test/"):
		self.path = path
		test_file = 'test-' + difficulty + '.txt'
		self.file_list = []
		with open(os.path.join(path, test_file), "r") as f:
			for line in f:
				line = line.replace("data/SNU-FILM/test/", img_data_path)
				line = line.strip()
				self.file_list.append(line.split(' '))	

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		name = self.file_list[index]
		I0_path = os.path.join(self.path, name[0])
		I1_path = os.path.join(self.path, name[1])
		I2_path = os.path.join(self.path, name[2])

		I0 = cv2.imread(I0_path)
		I1 = cv2.imread(I1_path)
		I2 = cv2.imread(I2_path)
		# BGR -> RBG
		I0 = I0[:, :, ::-1]
		I1 = I1[:, :, ::-1]
		I2 = I2[:, :, ::-1]

		I0 = torch.from_numpy(I0.copy()).permute(2, 0, 1) / 255.
		I1 = torch.from_numpy(I1.copy()).permute(2, 0, 1) / 255.
		I2 = torch.from_numpy(I2.copy()).permute(2, 0, 1) / 255. 

		padder = InputPadder(I0.shape, divisor=64)
		I0, I1, I2 = padder.pad(I0, I1, I2)
		return [I0, I1, I2]
	

if __name__ == "__main__":
	from torch.utils.data import DataLoader

	snudataset = SNUDataset(difficulty='hard')

	data_loader = DataLoader(snudataset, batch_size=1)

	for (I0, I1, I2) in data_loader:
		I0 = I0.cuda()
		I1 = I1.cuda()
		I2 = I2.cuda()

		# print(I0.size())
		# print(I1.size())
		# print(I2.size())

		# break