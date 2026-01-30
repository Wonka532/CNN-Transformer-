import torch
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import os
from models.network import FiberModeNet

MODEL_PATH = r"D:\桌面\optica\python\CNN+Transformer(24)\outputs_24modes\best_model.pth"
CSV_PATH = r"E:\training dataset val24\image_data.csv"
CONFIG_PATH = "config.yaml"
SAMPLE_INDEX = 34
def load_image_tensor(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image not found {image_path}")
            return None
        img = Image.open(image_path).convert('L')
        img = img.resize((256, 256))
        img_np = np.array(img) / 255.0
        h, w = img_np.shape
        cx, cy = h // 2, w // 2
        patches = [
            img_np[:cx, :cy], img_np[:cx, cy:],
            img_np[cx:, :cy], img_np[cx:, cy:]
        ]
        tensor = torch.from_numpy(np.array(patches)).float()
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error: {e}")
        return None
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading config...")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['num_modes'] = 24
    num_modes = 24
    model = FiberModeNet(config, physics_module=None).to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded (strict=False).")
        except Exception as e:
            print(f"Error: {e}")
            return
        model.eval()
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    row = df.iloc[SAMPLE_INDEX]
    if 'Labels_47dim' in row:
        gt_str = row['Labels_47dim']
    else:
        gt_str = row.iloc[0]
    if isinstance(gt_str, str):
        gt_values = np.fromstring(gt_str, sep=' ' if ' ' in gt_str else ',')
    else:
        gt_values = np.array(gt_str)
    image_path = row['Spatial_Path']
    input_tensor = load_image_tensor(image_path)
    if input_tensor is None: return
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_values = outputs['coeffs'].cpu().numpy()[0]

    gt_amps = gt_values[:num_modes]
    gt_betas_raw = gt_values[num_modes:]
    pred_amps = pred_values[:num_modes]
    pred_betas_raw = pred_values[num_modes:]
    gt_betas_full = np.insert(gt_betas_raw, 0, 1.0)
    pred_betas_full = np.insert(pred_betas_raw, 0, 1.0)
    mode_names = [
        "HE11_odd", "HE11_even",  # 1, 2
        "TE01",  # 3
        "TM01",  # 4
        "HE21_odd", "HE21_even",  # 5, 6
        "EH11_odd", "EH11_even",  # 7, 8
        "HE31_odd", "HE31_even",  # 9, 10
        "HE12_odd", "HE12_even",  # 11, 12
        "EH21_odd", "EH21_even",  # 13, 14
        "HE41_odd", "HE41_even",  # 15, 16
        "TE02",  # 17
        "TM02",  # 18
        "HE22_odd", "HE22_even",  # 19, 20
        "EH31_odd", "EH31_even",  # 21, 22
        "HE51_odd", "HE51_even"  # 23, 24
    ]
    # ==============================================================

    print(f"\n评估样本: {SAMPLE_INDEX}")
    header = f"{'模式':<12} | {'预测振幅':<10} {'真实振幅':<10} {'相差':<10} | {'预测映射值':<10} {'真实映射值':<10} {'相差':<10}"
    print(header)

    for i in range(num_modes):
        p_a, t_a = pred_amps[i], gt_amps[i]
        p_b, t_b = pred_betas_full[i], gt_betas_full[i]
        print(
            f"{mode_names[i]:<12} | {p_a:.5f}    {t_a:.5f}    {abs(p_a - t_a):.5f}    | {p_b:.5f}    {t_b:.5f}    {abs(p_b - t_b):.5f}")

    fidelity = np.dot(pred_amps, gt_amps) / (np.linalg.norm(pred_amps) * np.linalg.norm(gt_amps))
    print(f"振幅余弦相似度 : {fidelity:.6f}")

    pred_str = " ".join([f"{x:.6f}" for x in pred_values])
    print(f"预测输出字符串 : \n{pred_str}")


if __name__ == '__main__':
    main()