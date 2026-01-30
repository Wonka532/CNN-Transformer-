clear; clc; close all;
lambda0 = 1030e-9; 
num_modes = 24;             
radius = 62.5;              
Nx = 128;                   
spatial_window = 125;       
num_iterations = 800000;
folder_name = "E:\training dataset val24"; 
sep_char = '\';
% GRIN 光纤参数
profile_function = @build_GRIN; 
extra_params.ncore_diff = 0.0137; 
extra_params.alpha = 2.08; 
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end
%  2. 计算矢量模基底
lambda_um = lambda0 * 1e6; 
[epsilon, x, dx] = profile_function(lambda_um, Nx, spatial_window, radius, extra_params);
guess = sqrt(epsilon(round(Nx/2), round(Nx/2)));
boundary = '0000'; 

[phi_ex, neff_ex] = svmodes(lambda_um, guess, num_modes, dx, dx, epsilon, boundary, 'EX');
[phi_ey, neff_ey] = svmodes(lambda_um, guess, num_modes, dx, dx, epsilon, boundary, 'EY');

for k = 1:num_modes
    energy_ex = sum(sum(abs(phi_ex(:,:,k)).^2));
    if energy_ex > 0, phi_ex(:,:,k) = phi_ex(:,:,k) / sqrt(energy_ex); end
    energy_ey = sum(sum(abs(phi_ey(:,:,k)).^2));
    if energy_ey > 0, phi_ey(:,:,k) = phi_ey(:,:,k) / sqrt(energy_ey); end
end

save_mat_path = sprintf('%s%cmode_bases.mat', folder_name, sep_char);
phi_ex_real = real(phi_ex); phi_ex_imag = imag(phi_ex);
phi_ey_real = real(phi_ey); phi_ey_imag = imag(phi_ey);
save(save_mat_path, 'phi_ex_real', 'phi_ex_imag', 'phi_ey_real', 'phi_ey_imag', '-v7.3');

% 3. 数据记录
csv_filename = sprintf('%s%cimage_data.csv', folder_name, sep_char);
csv_header = {'Labels_47dim', 'Spatial_Path', 'FarField_Path'};
csv_data = cell(num_iterations, 3);

% 4. 生成数据
fprintf('开始生成 %d 组样本 (24 Modes)...\n', num_iterations);
t_start = tic;
for iter = 1:num_iterations
    if mod(iter, 1000) == 0
        elapsed = toc(t_start);
        avg_time = elapsed / iter;
        remain_time = avg_time * (num_iterations - iter);
        fprintf('进度: %d/%d | 剩余: %.1f s\n', iter, num_iterations, remain_time);
    end 
    % 生成物理系数
    raw_amplitudes = rand(1, num_modes); 
    amplitude_coef = raw_amplitudes / sqrt(sum(raw_amplitudes.^2));
    phase_raw = (rand(1, num_modes) * 2 * pi) - pi;
    % 计算标签 (保留相对相位)
    ref_phase = phase_raw(1); 
    phase_relative = angle(exp(1i * (phase_raw - ref_phase)));
    phase_mapped_all = (cos(phase_relative) + 1) / 2.0;
    phase_mapped_train = phase_mapped_all(2:end); 
    label_vec = [amplitude_coef, phase_mapped_train];
    coeff_str = sprintf('%.6f ', label_vec);
    coeff_str = strtrim(coeff_str); 
    % 物理场合成
    Ex_total = zeros(Nx, Nx);
    Ey_total = zeros(Nx, Nx);
    for k = 1:num_modes
        complex_coef = amplitude_coef(k) * exp(1i * phase_raw(k));
        Ex_total = Ex_total + complex_coef * phi_ex(:, :, k);
        Ey_total = Ey_total + complex_coef * phi_ey(:, :, k);
    end
    
    %  生成图像
    E_0deg = Ex_total;
    E_90deg = Ey_total;
    E_45deg = (Ex_total + Ey_total) / sqrt(2);
    E_n45deg = (Ex_total - Ey_total) / sqrt(2);
    spatial_combined = zeros(256, 256);
    spatial_combined(1:128, 1:128)     = abs(E_0deg).^2;
    spatial_combined(1:128, 129:256)   = abs(E_45deg).^2;
    spatial_combined(129:256, 1:128)   = abs(E_90deg).^2;
    spatial_combined(129:256, 129:256) = abs(E_n45deg).^2;
    FT_0deg = fftshift(fft2(E_0deg));
    FT_45deg = fftshift(fft2(E_45deg));
    FT_90deg = fftshift(fft2(E_90deg));
    FT_n45deg = fftshift(fft2(E_n45deg));
    freq_combined = zeros(256, 256);
    freq_combined(1:128, 1:128)     = abs(FT_0deg).^2;
    freq_combined(1:128, 129:256)   = abs(FT_45deg).^2;
    freq_combined(129:256, 1:128)   = abs(FT_90deg).^2;
    freq_combined(129:256, 129:256) = abs(FT_n45deg).^2;
    % 保存
    s_name = sprintf('s_%d.png', iter);
    w_name = sprintf('w_%d.png', iter); 
    full_s_path = sprintf('%s%c%s', folder_name, sep_char, s_name);
    full_w_path = sprintf('%s%c%s', folder_name, sep_char, w_name);
    imwrite(uint8(mat2gray(spatial_combined) * 255), full_s_path);
    imwrite(uint8(mat2gray(freq_combined) * 255), full_w_path);
    csv_data{iter, 1} = coeff_str;
    csv_data{iter, 2} = full_s_path;
    csv_data{iter, 3} = full_w_path;
end
csv_table = cell2table(csv_data, 'VariableNames', csv_header);
writetable(csv_table, csv_filename, 'Encoding', 'UTF-8');
