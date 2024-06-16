import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, smoothness_weight: float = 0.1):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    
    # Smoothness Loss の計算
    def smoothness_loss(flow):
        dx = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        dy = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        return torch.mean(dx) + torch.mean(dy)

    smoothness = smoothness_loss(pred_flow)
    loss = epe + smoothness_weight * smoothness

    return loss

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    
    # 学習率のスケジューリング
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.train.scheduler.step_size, gamma=args.train.scheduler.gamma)

    # ------------------
    #   Start training
    # ------------------
    model.train()
    best_loss = float('inf')
    start_epoch = 0

    # checkpointがあれば、そこから再開
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            flow = model(event_image) # [B, 2, 480, 640]
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow, smoothness_weight=args.train.smoothness_weight)
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # 学習率を更新
        scheduler.step()
        
        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # モデルの保存（最良の検証精度が更新された場合）
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, 'checkpoints/best_model.pth')
            print(f"Best model saved with loss {best_loss}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission.npy"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
