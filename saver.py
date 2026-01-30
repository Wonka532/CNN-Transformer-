import os
import torch
import csv
import time
class TrainingSaver:
    def __init__(self, output_dir, log_dir, resume=False):
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self.latest_ckpt_path = os.path.join(output_dir, 'latest_checkpoint.pth')
        self.best_ckpt_path = os.path.join(output_dir, 'best_model.pth')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        if not resume or not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Learning_Rate', 'Time_Sec', 'Timestamp'])
    def log_csv(self, epoch, train_loss, val_loss, lr):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}",
                f"{time.time():.2f}", time.strftime("%Y-%m-%d %H:%M:%S")
            ])
    def save_checkpoint(self, epoch, model, optimizer, scheduler, val_loss, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }
        torch.save(state, self.latest_ckpt_path)
        if is_best:
            torch.save(state, self.best_ckpt_path)
            print(f"[Saver] Best model saved (Val Loss: {val_loss:.6f})")
    def load_checkpoint(self, model, optimizer, scheduler):
        if not os.path.exists(self.latest_ckpt_path):
            print("[Saver] No checkpoint found. Starting from scratch.")
            return 0, float('inf')
        print(f"[Saver] Loading checkpoint from {self.latest_ckpt_path}...")
        try:
            checkpoint = torch.load(self.latest_ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            return start_epoch, checkpoint['val_loss']
        except Exception as e:
            print(f"[Saver] Error loading checkpoint: {e}. Starting from scratch.")
            return 0, float('inf')