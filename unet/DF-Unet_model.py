""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn.functional as F

class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=4, auxseg=False):
        super(SelFuseFeature, self).__init__()

        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True),
                                       )
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.

        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        grid_ = grid + 0.
        grid[..., 0] = 2 * grid_[..., 0] / (H - 1) - 1
        grid[..., 1] = 2 * grid_[..., 1] / (W - 1) - 1

        # features = []
        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))
        if self.auxseg:
            auxseg = self.auxseg_conv(x)
        else:
            auxseg = None

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return [select_x, auxseg]

class DF-Unet(nn.Module):
    def __init__(self, n_channels, n_classes=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        # Direct Field
        shift_n = 1
        auxseg = True
        selfeat = True
        num_class = 4
        self.ConvDf_1x1 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = SelFuseFeature(64, auxseg=auxseg, shift_n=shift_n)

        self.Conv_1x1 = nn.Conv2d(64, num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Direct Field
        df = self.ConvDf_1x1(x)
        # df = None
        if True:
            d2_auxseg = self.SelDF(x, df)
            x, auxseg = d2_auxseg[:2]
        else:
            auxseg = None
        logits = self.outc(x)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
