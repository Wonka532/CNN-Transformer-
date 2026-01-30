import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNBackbone(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )
        self.out_channels = base_channels * 8
    def forward(self, x):
        return self.features(x)
class FiberModeNet(nn.Module):
    def __init__(self, config, physics_module):
        super().__init__()
        self.physics = physics_module
        self.num_modes = config['data']['num_modes']  # 6
        # 1. 骨干网络
        self.cnn = CNNBackbone(in_channels=config['model']['cnn']['in_channels'])
        self.d_model = config['model']['transformer']['d_model']
        self.proj = nn.Conv2d(self.cnn.out_channels, self.d_model, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, self.d_model, 8, 8) * 0.05)
        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['model']['transformer']['nhead'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout'],
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['transformer']['num_layers'])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 3. 预测头
        self.output_dim = self.num_modes + (self.num_modes - 1)
        self.coeff_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, self.output_dim)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        feat = self.cnn(x)
        feat = self.proj(feat) + self.pos_embed
        feat_flat = feat.flatten(2).permute(0, 2, 1)
        enc_out = self.transformer(feat_flat)
        global_feat = self.avg_pool(enc_out.permute(0, 2, 1)).squeeze(-1)
        raw_out = self.coeff_head(global_feat)  # [B, 11]
        raw_amps = raw_out[:, :self.num_modes]
        raw_betas = raw_out[:, self.num_modes:]
        pred_amps = torch.sigmoid(raw_amps) + 1e-8
        pred_amps = F.normalize(pred_amps, p=2, dim=1)
        pred_betas = torch.sigmoid(raw_betas)
        pred_coeffs = torch.cat([pred_amps, pred_betas], dim=1)
        rec_nf, rec_ff = None, None
        if self.physics is not None:
            rec_nf, rec_ff = self.physics(pred_coeffs)
        return {
            'coeffs': pred_coeffs,  # 用于计算 MSE Loss
            'rec_nearfield': rec_nf,
            'rec_farfield': rec_ff
        }