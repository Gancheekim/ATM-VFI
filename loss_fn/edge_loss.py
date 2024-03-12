import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G45 = torch.tensor([[0., -2., -4.], [2., 0., -2.], [4., 2., 0.]])
        G135 = torch.tensor([[-4., -2., 0.], [-2., 0., 2.], [0., 2., 4.]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0), G45.unsqueeze(0), G135.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

        self.loss = nn.L1Loss()

    def _get_sobel(self, img, eps=1e-8):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + eps)
        return x
    
    def rgb2y(self, img):
        assert len(img.size()) == 4
        y = 0.299*img[:,0] + 0.587*img[:,1] + 0.114*img[:,2] 
        return y.unsqueeze(1)
    
    def forward(self, pred, gt):
        gt = self.rgb2y(gt.detach())
        pred = self.rgb2y(pred)
        gt = self._get_sobel(gt)
        pred = self._get_sobel(pred)
        return self.loss(pred, gt)