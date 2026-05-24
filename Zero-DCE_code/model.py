import torch
import torch.nn as nn


class SepConv3d(nn.Module):
	"""
	3D separable convolution as described in Zero-3DCE.
	Factorises a full 3D conv into:
	  - spatial  (1,3,3): learns spatial features within a frame
	  - temporal (3,1,1): learns temporal relationships across frames
	This halves parameters vs a full (3,3,3) conv and matches the paper architecture.
	"""
	def __init__(self, in_ch, out_ch, bias=True):
		super(SepConv3d, self).__init__()
		self.spatial  = nn.Conv3d(in_ch,  out_ch, (1,3,3), padding=(0,1,1), bias=bias)
		self.temporal = nn.Conv3d(out_ch, out_ch, (3,1,1), padding=(1,0,0), bias=bias)

	def forward(self, x):
		return self.temporal(self.spatial(x))


class SpatialAttention(nn.Module):
	"""
	Spatial attention module from Zero-3DCE.
	Generates a per-voxel attention map from channel-wise avg and max statistics,
	guiding the network to focus enhancement on dark/degraded regions.
	"""
	def __init__(self):
		super(SpatialAttention, self).__init__()
		self.conv    = nn.Conv3d(2, 1, (1,7,7), padding=(0,3,3), bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
		return x * attn


class Zero3DCE(nn.Module):
	"""
	Zero-3DCE: Zero-Reference 3D Curve Estimation for Video Low-Light Enhancement.

	Key differences from Zero-DCE:
	- SepConv3d (separable 3D convolutions) instead of Conv2D
	- SpatialAttention guides enhancement to regions that need it
	- Processes video clips (B, 3, T, H, W) for temporal consistency
	- 3D cross-channel curve estimation (72 = 8 iters x 9 params)
	"""

	def __init__(self):
		super(Zero3DCE, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		nf = 32

		self.e_conv1 = SepConv3d(3,    nf)
		self.e_conv2 = SepConv3d(nf,   nf)
		self.e_conv3 = SepConv3d(nf,   nf)
		self.e_conv4 = SepConv3d(nf,   nf)
		self.attn    = SpatialAttention()       # applied after deepest encoder layer
		self.e_conv5 = SepConv3d(nf*2, nf)
		self.e_conv6 = SepConv3d(nf*2, nf)
		# 8 iterations × 18 params (9 shadow + 9 highlight) = 144 output channels
		self.e_conv7 = SepConv3d(nf*2, 144)

	def apply_3d_curve(self, x, r):
		# x: (B, 3, T, H, W)   r: (B, 18, T, H, W) — shadow + highlight params
		B, _, T, H, W = x.shape
		r_s = r[:, :9].view(B, 3, 3, T, H, W)   # shadow coefficients
		r_h = r[:, 9:].view(B, 3, 3, T, H, W)   # highlight coefficients
		shadow_basis    = x * (1.0 - x) ** 2     # peaks at x≈0.33, lifts shadows
		highlight_basis = (x ** 2) * (1.0 - x)   # peaks at x≈0.67, adjusts midtones/highlights
		shadow_delta    = (r_s * shadow_basis.unsqueeze(1)).sum(dim=2)
		highlight_delta = (r_h * highlight_basis.unsqueeze(1)).sum(dim=2)
		return x + shadow_delta + highlight_delta

	def forward(self, x):
		# x: (B, 3, T, H, W)
		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x4 = self.attn(x4)                             # spatial attention
		x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
		x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 18, dim=1)

		x = self.apply_3d_curve(x, r1)
		x = self.apply_3d_curve(x, r2)
		x = self.apply_3d_curve(x, r3)
		enhance_1 = self.apply_3d_curve(x, r4)
		x = self.apply_3d_curve(enhance_1, r5)
		x = self.apply_3d_curve(x, r6)
		x = self.apply_3d_curve(x, r7)
		enhance   = self.apply_3d_curve(x, r8)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8], 1)
		return enhance_1, enhance, r
