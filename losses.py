import torch
import torch.nn as nn
class CombinedLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
    def normalize_img(self, img):
        max_val = img.flatten(1).max(dim=1)[0]
        max_val = max_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return img / (max_val + 1e-8)
    def forward(self, preds, targets):
        # 1. 系数 Loss
        loss_coeff = self.mse(preds['coeffs'], targets['coeffs'])
        # 2. 物理 Loss
        loss_nf = torch.tensor(0.0, device=preds['coeffs'].device)
        loss_ff = torch.tensor(0.0, device=preds['coeffs'].device)
        if self.weights.get('nearfield', 0) > 0:
            rec_nf_norm = self.normalize_img(preds['rec_nearfield'])
            gt_nf_norm = self.normalize_img(targets['image'])
            loss_nf = self.mse(rec_nf_norm, gt_nf_norm)
        if self.weights.get('farfield', 0) > 0:
            rec_ff_norm = self.normalize_img(preds['rec_farfield'])
            gt_ff_norm = self.normalize_img(targets['gt_farfield'])
            loss_ff = self.mse(rec_ff_norm, gt_ff_norm)

        total_loss = (self.weights['coeff'] * loss_coeff +
                      self.weights.get('nearfield', 0) * loss_nf +
                      self.weights.get('farfield', 0) * loss_ff)
        return total_loss, {
            'loss_coeff': loss_coeff.item(),
            'loss_nf': loss_nf.item(),
            'loss_ff': loss_ff.item()
        }