import torch
import torch.nn as nn


class Zero3DCE(nn.Module):
	"""
	Zero-3DCE: Zero-Reference 3D Deep Curve Estimation for Video Low-Light Enhancement.

	Processes video clips of shape (B, 3, T, H, W).
	Conv3D layers with temporal kernels capture motion context across frames,
	preventing the per-frame flicker that a Conv2D model would produce.
	"""

	def __init__(self, clip_len=8):
		super(Zero3DCE, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		nf = 32

		# (1,3,3) kernel: spatial only — used at input/output to avoid edge artefacts
		# (3,3,3) kernel: spatiotemporal — used in middle layers for temporal context
		self.e_conv1 = nn.Conv3d(3,    nf,   (1,3,3), 1, (0,1,1), bias=True)
		self.e_conv2 = nn.Conv3d(nf,   nf,   (3,3,3), 1, (1,1,1), bias=True)
		self.e_conv3 = nn.Conv3d(nf,   nf,   (3,3,3), 1, (1,1,1), bias=True)
		self.e_conv4 = nn.Conv3d(nf,   nf,   (3,3,3), 1, (1,1,1), bias=True)
		self.e_conv5 = nn.Conv3d(nf*2, nf,   (1,3,3), 1, (0,1,1), bias=True)
		self.e_conv6 = nn.Conv3d(nf*2, nf,   (1,3,3), 1, (0,1,1), bias=True)
		# 8 iterations × 9 cross-channel curve params = 72 output channels
		self.e_conv7 = nn.Conv3d(nf*2, 72,   (1,3,3), 1, (0,1,1), bias=True)

	def apply_3d_curve(self, x, r):
		# x: (B, 3, T, H, W)
		# r: (B, 9, T, H, W) — 3×3 cross-channel curve matrix per voxel
		# delta[out_c] = sum_{in_c} r[out_c, in_c] * (x[in_c]^2 - x[in_c])
		B, _, T, H, W = x.shape
		r = r.view(B, 3, 3, T, H, W)          # (B, out_ch, in_ch, T, H, W)
		quadratic = x * x - x                  # negative for x in (0,1) → brightens with +r
		delta = (r * quadratic.unsqueeze(1)).sum(dim=2)  # (B, 3, T, H, W)
		return x + delta

	def forward(self, x):
		# x: (B, 3, T, H, W)
		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
		x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 9, dim=1)

		x = self.apply_3d_curve(x, r1)
		x = self.apply_3d_curve(x, r2)
		x = self.apply_3d_curve(x, r3)
		enhance_1 = self.apply_3d_curve(x, r4)
		x = self.apply_3d_curve(enhance_1, r5)
		x = self.apply_3d_curve(x, r6)
		x = self.apply_3d_curve(x, r7)
		enhance  = self.apply_3d_curve(x, r8)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8], 1)
		return enhance_1, enhance, r
