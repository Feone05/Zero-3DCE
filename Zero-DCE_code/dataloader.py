import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path

random.seed(1143)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


class VideoClipLoader(data.Dataset):
	"""
	Loads fixed-length clips of frames for video enhancement training.

	Folder layout (preferred — one sub-folder per video):
	    train_data/
	        video_001/  frame_0001.jpg  frame_0002.jpg ...
	        video_002/  frame_0001.jpg  ...

	Flat layout (fallback — random clips sampled from all images):
	    train_data/  img1.jpg  img2.jpg  ...
	    Each image is repeated clip_len times to form a single-frame "clip".
	"""

	def __init__(self, data_path, clip_len=8, size=256):
		self.size     = size
		self.clip_len = clip_len
		self.clips    = []

		root = Path(data_path)
		if not root.exists():
			raise FileNotFoundError(f"Training data folder not found: {root.resolve()}\nCreate it and add low-light images or video frame sub-folders.")

		subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

		if subdirs:
			for seq_dir in subdirs:
				frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
				if not frames:
					continue
				if len(frames) >= clip_len:
					stride = max(1, clip_len // 2)
					for i in range(0, len(frames) - clip_len + 1, stride):
						self.clips.append(frames[i:i + clip_len])
				else:
					# Pad short sequences by repeating the last frame
					padded = frames + [frames[-1]] * (clip_len - len(frames))
					self.clips.append(padded)
		else:
			# Flat layout: treat each image as a clip of identical frames
			images = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
			for img in images:
				self.clips.append([img] * clip_len)

		random.shuffle(self.clips)
		print(f"Total video clips: {len(self.clips)}  (clip_len={clip_len})")

	def _load_frame(self, path):
		img = Image.open(path).convert('RGB')
		img = img.resize((self.size, self.size), Image.LANCZOS)
		arr = np.asarray(img, dtype=np.float32) / 255.0
		return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

	def __getitem__(self, index):
		frames = [self._load_frame(p) for p in self.clips[index]]
		# (T, 3, H, W) → (3, T, H, W)
		return torch.stack(frames, dim=0).permute(1, 0, 2, 3)

	def __len__(self):
		return len(self.clips)
