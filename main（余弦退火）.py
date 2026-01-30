import torch
from torch.utils.data import DataLoader, random_split
import yaml
import os
import math
from tqdm import tqdm
from data.dataset import FiberModeDataset
from data.bases import ModeBasesLoader
from models.network import FiberModeNet
from models.physics import PhysicsReconstructionModule
from utils.losses import CombinedLoss
from utils.saver import TrainingSaver

LOG_DIR = r"E:\training dataset 24"
RESUME_TRAINING = True
# 余弦退火
def get_lr_lambda(config):
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['num_epochs']
    min_lr_ratio = config['training']['min_lr'] / config['training']['learning_rate']
    warmup_start_ratio = 0.1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * float(epoch) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return lr_lambda
def train():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 1. 数据
    dataset = FiberModeDataset(config['data']['csv_path'])
    total_size = len(dataset)
    train_size = int(total_size * config['data']['train_ratio'])
    val_size = int(total_size * config['data']['val_ratio'])
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config['data']['batch_size'], shuffle=True,
                              num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['data']['batch_size'], shuffle=False)
    # 2. 模型
    bases_loader = ModeBasesLoader(config['data']['mode_bases_path'], device=device)
    phi_ex, phi_ey = bases_loader.get_bases()
    physics_module = PhysicsReconstructionModule(phi_ex, phi_ey).to(device)
    model = FiberModeNet(config, physics_module).to(device)
    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config['training']['learning_rate']),
                                  weight_decay=float(config['training']['weight_decay']))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(config))
    criterion = CombinedLoss(config['training']['loss_weights'])
    # 4. Saver
    output_dir = config['project']['output_dir']
    saver = TrainingSaver(output_dir=output_dir, log_dir=LOG_DIR, resume=RESUME_TRAINING)
    start_epoch = 0
    best_val_loss = float('inf')
    if RESUME_TRAINING:
        loaded_epoch, loaded_val_loss = saver.load_checkpoint(model, optimizer, lr_scheduler)
        start_epoch = loaded_epoch
        if start_epoch > 0:
            best_val_loss = loaded_val_loss
    # 5. 训练循环
    physics_start_epoch = config['training']['physics_start_epoch']
    num_epochs = config['training']['num_epochs']
    print(f"Training Start: Epoch {start_epoch} -> {num_epochs} (Strategy: Cosine Annealing)")
    try:
        for epoch in range(start_epoch, num_epochs):
            if epoch >= physics_start_epoch:
                criterion.weights['nearfield'] = config['training']['loss_weights']['nearfield_target']
                criterion.weights['farfield'] = config['training']['loss_weights']['farfield_target']
                physics_status = "ON"
            else:
                criterion.weights['nearfield'] = 0.0
                criterion.weights['farfield'] = 0.0
                physics_status = "OFF"
            # Train
            model.train()
            train_loss_accum = 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [LR={current_lr:.2e}|Phys={physics_status}]")

            for batch in pbar:
                inputs = batch['image'].to(device)
                targets = {
                    'image': inputs,
                    'gt_farfield': batch['gt_farfield'].to(device),
                    'coeffs': batch['coeffs'].to(device)
                }
                optimizer.zero_grad()
                preds = model(inputs)
                loss, loss_dict = criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()
                pbar.set_postfix({'Loss': loss.item(), 'NF': loss_dict.get('loss_nf', 0)})
            lr_scheduler.step()
            # Val
            model.eval()
            val_loss_accum = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device)
                    targets = {
                        'image': inputs,
                        'gt_farfield': batch['gt_farfield'].to(device),
                        'coeffs': batch['coeffs'].to(device)
                    }
                    preds = model(inputs)
                    loss, _ = criterion(preds, targets)
                    val_loss_accum += loss.item()
            avg_train = train_loss_accum / len(train_loader)
            avg_val = val_loss_accum / len(val_loader)
            print(f"Epoch {epoch + 1} Summary - Train: {avg_train:.6f} - Val: {avg_val:.6f}")
            # --- Save ---
            saver.log_csv(epoch, avg_train, avg_val, current_lr)
            is_best = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
            saver.save_checkpoint(epoch, model, optimizer, lr_scheduler, avg_val, is_best)
    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted by user. Checkpoint saved.")
    print("Training process finished.")
if __name__ == '__main__':
    train()