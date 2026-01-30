import torch
import scipy.io
import numpy as np
import h5py
class ModeBasesLoader:
    def __init__(self, mat_path, device='cpu'):
        self.mat_path = mat_path
        self.device = device
        self.phi_ex = None
        self.phi_ey = None
        self._load()
    def _load(self):
        try:
            data = scipy.io.loadmat(self.mat_path)
            is_v73 = False
        except NotImplementedError:
            print("Detected MATLAB v7.3 format, switching to HDF5 reader.")
            data = h5py.File(self.mat_path, 'r')
            is_v73 = True
        def process_field(real_key, imag_key):
            if is_v73:
                real = np.array(data[real_key])
                imag = np.array(data[imag_key])
                real = real.transpose(0, 2, 1)
                imag = imag.transpose(0, 2, 1)
            else:
                real = data[real_key].transpose(2, 0, 1)
                imag = data[imag_key].transpose(2, 0, 1)
            complex_tensor = torch.from_numpy(real + 1j * imag).cfloat()
            energy = (complex_tensor.abs() ** 2).sum(dim=(-2, -1), keepdim=True)
            complex_tensor = complex_tensor / (torch.sqrt(energy) + 1e-8)
            return complex_tensor.unsqueeze(0)  # 增加 Batch 维 -> [1, 16, 128, 128]
        self.phi_ex = process_field('phi_ex_real', 'phi_ex_imag').to(self.device)
        self.phi_ey = process_field('phi_ey_real', 'phi_ey_imag').to(self.device)
        if is_v73:
            data.close()
        print("Mode bases loaded successfully.")
    def get_bases(self):
        return self.phi_ex, self.phi_ey