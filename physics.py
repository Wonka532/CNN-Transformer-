import torch
import torch.nn as nn
import torch.fft
class PhysicsReconstructionModule(nn.Module):
    def __init__(self, phi_ex, phi_ey):
        super().__init__()
        self.register_buffer('phi_ex', phi_ex)
        self.register_buffer('phi_ey', phi_ey)
    def forward(self, pred_coeffs):
        B = pred_coeffs.shape[0]
        n_modes = self.phi_ex.shape[1]
        amps = pred_coeffs[:, :n_modes]
        betas_partial = pred_coeffs[:, n_modes:]
        ref_beta = torch.ones(B, 1, device=pred_coeffs.device)
        betas_full = torch.cat([ref_beta, betas_partial], dim=1)
        cos_theta = 2.0 * betas_full - 1.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        thetas_recon = torch.acos(cos_theta)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        real_part = amps * cos_theta
        imag_part = amps * sin_theta
        complex_coeffs = torch.complex(real_part, imag_part)  # [B, 6]
        coeffs_bc = complex_coeffs.unsqueeze(-1).unsqueeze(-1)
        E_total_x = torch.sum(coeffs_bc * self.phi_ex, dim=1)
        E_total_y = torch.sum(coeffs_bc * self.phi_ey, dim=1)
        def get_intensity(field):
            return field.real.pow(2) + field.imag.pow(2)
        I_0 = get_intensity(E_total_x)
        I_90 = get_intensity(E_total_y)
        I_45 = 0.5 * get_intensity(E_total_x + E_total_y)
        I_n45 = 0.5 * get_intensity(E_total_x - E_total_y)
        rec_near_fields = torch.stack([I_0, I_45, I_90, I_n45], dim=1)
        FF_x = torch.fft.fftshift(torch.fft.fft2(E_total_x, norm='ortho'), dim=(-2, -1))
        FF_y = torch.fft.fftshift(torch.fft.fft2(E_total_y, norm='ortho'), dim=(-2, -1))
        rec_far_field = (get_intensity(FF_x) + get_intensity(FF_y)).unsqueeze(1)
        return rec_near_fields, rec_far_field