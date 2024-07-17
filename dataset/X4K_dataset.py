import glob
import numpy as np
import torch
import cv2
import random
import torch.utils.data as data
import os

def make_2D_dataset_X_Train(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for scene_path in sorted(glob.glob(os.path.join(dir, '*', ''))):
        sample_paths = sorted(glob.glob(os.path.join(scene_path, '*', '')))
        for sample_path in sample_paths:
            frame65_list = []
            for frame in sorted(glob.glob(os.path.join(sample_path, '*.png'))):
                frame65_list.append(frame)
            framesPath.append(frame65_list)

    print("The number of total training samples : {} which has 65 frames each.".format(
        len(framesPath)))  ## 4408 folders which have 65 frames each
    return framesPath

def make_2D_dataset_X_Test(dir, multiple, t_step_size):
    """ make [I0,I1,It,t,scene_folder] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [type1,type2,type3,...]
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
            for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                if idx == len(frame_folder) - 1:
                    break
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                    I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                    I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                    I0I1It_paths.append(t[mul])
                    I0I1It_paths.append(scene_folder.split(os.path.join(dir, ''))[-1])  # type1/scene1
                    testPath.append(I0I1It_paths)
    return testPath


def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # [T, H, W, C] -> [T, C, H, W]
    imgIn = torch.from_numpy(imgIn.copy()).permute(0,3,1,2) / 255.0
    return imgIn


def frames_loader_train(random_crop, candidate_frames, frameRange, patch_size=512, img_ch=3):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        frame = frame[:, :, ::-1] # BGR->RGB
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if random_crop:  ## random crop
        ps = patch_size
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [0,1] normalized """
    frames = RGBframes_np2Tensor(frames, img_ch)
    return frames


def frames_loader_test(I0I1It_Path, validation, img_ch):
    frames = []
    for path in I0I1It_Path:
        frame = cv2.imread(path)
        frame = frame[:, :, ::-1] # BGR->RGB
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    if validation:
        ps = 512
        ix = (iw - ps) // 2
        iy = (ih - ps) // 2
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, img_ch)
    return frames


class X_Train(data.Dataset):
    def __init__(self, train_data_path, max_t_step_size=32, min_t_step_size=8, random_crop=False, patch_size=512, img_ch=3):
        self.train_data_path = train_data_path
        self.random_crop = random_crop
        self.patch_size = patch_size
        self.img_ch = img_ch
        self.max_t_step_size = max_t_step_size
        self.min_t_step_size = min_t_step_size

        self.framesPath = make_2D_dataset_X_Train(self.train_data_path)
        self.nScenes = len(self.framesPath)

        # Raise error if no images found in train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.train_data_path + "\n"))

    def __getitem__(self, idx):
        t_step_size = random.randint(self.min_t_step_size, self.max_t_step_size)

        candidate_frames = self.framesPath[idx]
        firstFrameIdx = random.randint(0, (64 - t_step_size))
        interIdx = t_step_size // 2
        interFrameIdx = firstFrameIdx + interIdx  # absolute index

        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]

        frames = frames_loader_train(self.random_crop, candidate_frames, frameRange, patch_size=self.patch_size, img_ch=self.img_ch) # including "np2Tensor [-1,1] normalized"

        # frames: [T, C, H, W]
        img0 = frames[0]
        img1 = frames[1]
        gt = frames[2]
        return [img0, gt, img1]

    def __len__(self):
        return self.nScenes
    

class X_Test(data.Dataset):
    def __init__(self, data_path, multiple=2, validation=True, img_ch=3):
        self.multiple = multiple
        self.validation = validation
        self.data_path = data_path
        self.img_ch = img_ch
        if validation:
            self.testPath = make_2D_dataset_X_Test(self.data_path, multiple, t_step_size=32)
        else:  ## test
            self.testPath = make_2D_dataset_X_Test(self.data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            if validation:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))
            else:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, _, _ = self.testPath[idx]
        I0I1It_Path = [I0, I1, It]

        frames = frames_loader_test(I0I1It_Path, self.validation, self.img_ch)
        # including "np2Tensor [-1,1] normalized"

        # frames: [T, C, H, W]
        img0 = frames[0]
        img1 = frames[1]
        gt = frames[2]
        return [img0, gt, img1]

    def __len__(self):
        return self.nIterations

    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = X_Train(train_data_path="/home/kim/Desktop/ssd/X4K1000FPS/train", 
                      max_t_step_size=32)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for (im1, im2, im3) in train_loader:
        print(im1.size())
        print(im2.size())
        print(im3.size())
        print(torch.min(im1))
        print(torch.max(im1))
        break


    dataset = X_Test(data_path="/home/kim/Desktop/ssd/X4K1000FPS/test")
    val_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for (im1, im2, im3) in val_loader:
        print(im1.size())
        print(im2.size())
        print(im3.size())
        print(torch.min(im1))
        print(torch.max(im1))
        break