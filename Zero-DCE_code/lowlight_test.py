"""
Zero-3DCE inference for video enhancement.

Supported input layouts:
  A) Folder of frame sequences:
       test_data/
           video_001/  frame001.jpg  frame002.jpg ...
           video_002/  ...
     → enhanced frames saved to result/video_001/, result/video_002/, ...

  B) Flat folder of images (treated as a single-frame sequence):
       test_data/DICM/  img1.jpg  img2.jpg ...
     → enhanced frames saved to result/DICM/
"""

import torch
import torchvision
import os
import time
import model
import numpy as np
from PIL import Image
from pathlib import Path

CLIP_LEN = 8           # must match training
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_frame(path, device, size=None):
	img = Image.open(path).convert('RGB')
	if size is not None:
		img = img.resize(size, Image.LANCZOS)
	t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
	return t.permute(2, 0, 1).to(device)   # (3, H, W)


def enhance_sequence(frame_paths, out_dir, net, device):
	"""Enhance a list of frame paths and save results to out_dir."""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Pad to a multiple of CLIP_LEN by repeating the last frame
	frames = list(frame_paths)
	pad = (CLIP_LEN - len(frames) % CLIP_LEN) % CLIP_LEN
	frames += [frames[-1]] * pad

	# Use the first frame's size as the common clip resolution
	ref_size = Image.open(frames[0]).size  # (W, H)

	with torch.no_grad():
		for start in range(0, len(frames), CLIP_LEN):
			clip_paths = frames[start:start + CLIP_LEN]
			clip = torch.stack([load_frame(p, device, size=ref_size) for p in clip_paths], dim=1)  # (3, T, H, W)
			clip = clip.unsqueeze(0)  # (1, 3, T, H, W)

			t0 = time.time()
			_, enhanced, _ = net(clip)
			elapsed = time.time() - t0

			# Save only real frames (not padding)
			real = len(clip_paths) - (pad if start + CLIP_LEN >= len(frames) else 0)
			for i in range(real):
				src_name = Path(clip_paths[i]).name
				out_path = out_dir / src_name
				torchvision.utils.save_image(enhanced[0, :, i], str(out_path))
				print(f"  {clip_paths[i]}  ->  {out_path}  ({elapsed/real:.2f}s/frame)")


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	net = model.Zero3DCE().to(device)
	net.load_state_dict(torch.load('snapshots/Epoch99_3dce_video.pth', map_location=device))
	net.eval()

	test_root = Path('data/test_data')
	result_root = Path('data/result')

	for subdir in sorted(test_root.iterdir()):
		if not subdir.is_dir():
			continue
		frame_paths = sorted(p for p in subdir.rglob('*') if p.suffix.lower() in IMG_EXTS)
		if not frame_paths:
			continue
		print(f"\nProcessing {subdir.name}  ({len(frame_paths)} frames)")
		out_dir = result_root / subdir.name
		enhance_sequence(frame_paths, out_dir, net, device)
