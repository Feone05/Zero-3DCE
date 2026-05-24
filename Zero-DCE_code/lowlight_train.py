import torch
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def train(config):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	net = model.Zero3DCE().to(device)
	net.apply(weights_init)

	if config.load_pretrain:
		net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))

	dataset = dataloader.VideoClipLoader(
		config.lowlight_images_path,
		clip_len=config.clip_len,
		size=config.img_size)
	loader = torch.utils.data.DataLoader(
		dataset, batch_size=config.train_batch_size,
		shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color    = Myloss.L_color()
	L_spa      = Myloss.L_spa()
	L_exp      = Myloss.L_exp(16, 0.6)
	L_TV       = Myloss.L_TV()
	L_temporal = Myloss.L_temporal()
	L_sharp    = Myloss.L_sharp()
	L_contrast = Myloss.L_contrast()
	L_msssim   = Myloss.L_MSSSIM()
	L_edge     = Myloss.L_edge()

	optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	net.train()

	for epoch in range(config.num_epochs):
		for iteration, clip in enumerate(loader):
			# clip: (B, 3, T, H, W)
			clip = clip.to(device)
			_, enhanced, A = net(clip)

			# Flatten temporal dim for 2D losses: (B,3,T,H,W) → (B*T,3,H,W)
			B, C, T, H, W = enhanced.shape
			enh_2d  = enhanced.permute(0,2,1,3,4).reshape(B*T, C, H, W)
			clip_2d = clip.permute(0,2,1,3,4).reshape(B*T, C, H, W)
			# Flatten r for TV loss: (B,72,T,H,W) → (B*T,72,H,W)
			r_2d = A.permute(0,2,1,3,4).reshape(B*T, A.shape[1], H, W)

			loss = (200 * L_TV(r_2d)
				+       torch.mean(L_spa(enh_2d, clip_2d))
				+ 5   * torch.mean(L_color(enh_2d))
				+ 10  * torch.mean(L_exp(enh_2d))
				+ 20  * L_temporal(enhanced)
				+ 8   * L_sharp(enh_2d, clip_2d)
				+ 4   * L_contrast(enh_2d)
				+ 10  * L_msssim(enh_2d, clip_2d)      # structural preservation (MS-SSIM)
				+ 6   * L_edge(enh_2d, clip_2d))        # edge fidelity (Laplacian)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
			optimizer.step()

			if (iteration + 1) % config.display_iter == 0:
				print(f"Epoch {epoch}  iter {iteration+1}  loss {loss.item():.4f}")
			if (iteration + 1) % config.snapshot_iter == 0:
				torch.save(net.state_dict(), config.snapshots_folder + f"Epoch{epoch}.pth")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lowlight_images_path', type=str, default='data/train_data/')
	parser.add_argument('--lr',               type=float, default=0.0001)
	parser.add_argument('--weight_decay',     type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm',   type=float, default=0.1)
	parser.add_argument('--num_epochs',       type=int,   default=200)
	parser.add_argument('--train_batch_size', type=int,   default=4)
	parser.add_argument('--num_workers',      type=int,   default=4)
	parser.add_argument('--clip_len',         type=int,   default=8)
	parser.add_argument('--img_size',         type=int,   default=256)
	parser.add_argument('--display_iter',     type=int,   default=10)
	parser.add_argument('--snapshot_iter',    type=int,   default=10)
	parser.add_argument('--snapshots_folder', type=str,   default='snapshots/')
	parser.add_argument('--load_pretrain',    type=bool,  default=False)
	parser.add_argument('--pretrain_dir',     type=str,   default='snapshots/Epoch99.pth')

	config = parser.parse_args()
	os.makedirs(config.snapshots_folder, exist_ok=True)
	train(config)
