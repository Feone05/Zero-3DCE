import torch
import torch.nn as nn
import torch.nn.functional as F


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        # x: (B, 3, H, W)
        mean_rgb = torch.mean(x, [2,3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        return torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2), 0.5)


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left  = torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        # org, enhance: (B, 3, H, W)
        org_mean     = torch.mean(org,     1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        dev = org_pool.device
        weight_diff = torch.max(
            torch.FloatTensor([1]).to(dev) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(dev),
                                                              torch.FloatTensor([0]).to(dev)),
            torch.FloatTensor([0.5]).to(dev))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(dev)), enhance_pool - org_pool)

        D_org_left    = F.conv2d(org_pool,     self.weight_left,  padding=1)
        D_org_right   = F.conv2d(org_pool,     self.weight_right, padding=1)
        D_org_up      = F.conv2d(org_pool,     self.weight_up,    padding=1)
        D_org_down    = F.conv2d(org_pool,     self.weight_down,  padding=1)
        D_enh_left    = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        D_enh_right   = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enh_up      = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        D_enh_down    = F.conv2d(enhance_pool, self.weight_down,  padding=1)

        D_left  = torch.pow(D_org_left  - D_enh_left,  2)
        D_right = torch.pow(D_org_right - D_enh_right, 2)
        D_up    = torch.pow(D_org_up    - D_enh_up,    2)
        D_down  = torch.pow(D_org_down  - D_enh_down,  2)
        return D_left + D_right + D_up + D_down


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        # x: (B, 3, H, W)
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2))


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # x: (B, C, H, W)
        batch_size = x.size(0)
        h_x, w_x = x.size(2), x.size(3)
        count_h = (h_x - 1) * x.size(3)
        count_w =  h_x      * (w_x - 1)
        h_tv = torch.pow(x[:,:,1:,:]  - x[:,:,:h_x-1,:], 2).sum()
        w_tv = torch.pow(x[:,:,:,1:]  - x[:,:,:,:w_x-1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size


class L_temporal(nn.Module):
    """Temporal consistency loss — penalises luminance flicker between adjacent frames."""
    def __init__(self):
        super(L_temporal, self).__init__()

    def forward(self, enhanced):
        # enhanced: (B, 3, T, H, W)
        lum = enhanced.mean(dim=1)                          # (B, T, H, W)
        diff = lum[:, 1:, :, :] - lum[:, :-1, :, :]       # (B, T-1, H, W)
        return torch.mean(diff ** 2)


class L_sharp(nn.Module):
    """
    Sharpness preservation loss.
    Penalises when enhanced edge strength is weaker than the original input,
    ensuring object boundaries stay crisp after brightening.
    """
    def __init__(self):
        super(L_sharp, self).__init__()

    def forward(self, enhanced, original):
        # enhanced, original: (B, 3, H, W)
        def grad_mag(x):
            dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])   # (B,3,H,W-1)
            dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])   # (B,3,H-1,W)
            return (dx[:, :, :-1, :] + dy[:, :, :, :-1]) / 2    # (B,3,H-1,W-1)

        enh_g = grad_mag(enhanced)
        org_g = grad_mag(original)
        # Only penalise when enhanced is blurrier than original
        return torch.mean(F.relu(org_g - enh_g))


class L_contrast(nn.Module):
    """
    Global contrast loss.
    Maximises pixel variance across the image so objects stand out from their
    background, improving detectability in enhanced frames.
    """
    def __init__(self):
        super(L_contrast, self).__init__()

    def forward(self, enhanced):
        # enhanced: (B, 3, H, W)
        mean     = enhanced.mean(dim=[2, 3], keepdim=True)
        variance = torch.pow(enhanced - mean, 2).mean()
        return -variance   # negate: minimising this maximises contrast


class L_MSSSIM(nn.Module):
    """
    Multi-Scale SSIM loss as used in Zero-3DCE (Tatana et al. 2025).
    Measures structural similarity at 3 scales; 1 - MS-SSIM gives a loss
    that penalises blur and structural distortion caused by brightening.
    No external dependencies — SSIM computed with avg-pool approximation.
    """
    def __init__(self, scales=3, window_size=11):
        super(L_MSSSIM, self).__init__()
        self.scales      = scales
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x, y):
        pad = self.window_size // 2
        k   = self.window_size
        mu_x  = F.avg_pool2d(x,   k, stride=1, padding=pad)
        mu_y  = F.avg_pool2d(y,   k, stride=1, padding=pad)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sx  = F.avg_pool2d(x*x, k, stride=1, padding=pad) - mu_x2
        sy  = F.avg_pool2d(y*y, k, stride=1, padding=pad) - mu_y2
        sxy = F.avg_pool2d(x*y, k, stride=1, padding=pad) - mu_xy
        num = (2*mu_xy + self.C1) * (2*sxy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sx + sy + self.C2)
        return (num / den).mean()

    def forward(self, enhanced, original):
        # enhanced, original: (B, 3, H, W)
        loss = 0.0
        x, y = enhanced, original
        for _ in range(self.scales):
            loss += 1.0 - self._ssim(x, y)
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        return loss / self.scales


class L_edge(nn.Module):
    """
    Laplacian edge loss as used in Zero-3DCE (Tatana et al. 2025).
    Penalises differences in edge maps between enhanced and original,
    preserving structural detail critical for object detection.
    """
    def __init__(self):
        super(L_edge, self).__init__()
        kernel = torch.FloatTensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
        self.register_buffer('laplacian', kernel)

    def forward(self, enhanced, original):
        # enhanced, original: (B, 3, H, W)
        enh_lum = enhanced.mean(dim=1, keepdim=True)
        org_lum = original.mean(dim=1, keepdim=True)
        enh_edge = F.conv2d(enh_lum, self.laplacian, padding=1)
        org_edge = F.conv2d(org_lum, self.laplacian, padding=1)
        return torch.mean((enh_edge - org_edge) ** 2)
