import argparse
import json
import math
import os
import re
import time
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 用于监控模型性能的工具函数
def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_gpu_memory_usage(device):
    """获取 GPU 显存使用情况（MB）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        return allocated, reserved
    return 0, 0

def print_model_info(model, device):
    """打印模型信息"""
    total_params, trainable_params = count_parameters(model)
    allocated, reserved = get_gpu_memory_usage(device)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Memory Reserved: {reserved:.2f} MB")
    print("="*60 + "\n")

def range_loss(pred_range, target_range, beta=0.1):
    loss = F.smooth_l1_loss(pred_range, target_range, beta=beta, reduction='mean')
    return loss

def angle_loss_cos_sin(pred_xy, target_vec, beta=0.1):

    pred_xy = F.normalize(pred_xy, dim=-1)
    target_vec = F.normalize(target_vec, dim=-1)

    loss = F.smooth_l1_loss(pred_xy, target_vec, beta=beta, reduction='mean')
    return loss

def wrapped_range_error(pred_range, target_range):

    pred_range = pred_range.float()
    target_range = target_range.float()

    #print(pred_range, target_range)
    diff = pred_range - target_range

    mse = (diff ** 2).mean().item() 
    mae = diff.abs().mean().item()   
    return mae, mse

def wrapped_angle_error_deg(pred_xy, target_vec):

    p = F.normalize(pred_xy, dim=-1)
    t = F.normalize(target_vec, dim=-1)

    cos_d = (p * t).sum(dim=-1).clamp(-1.0, 1.0)
    sin_d = p[:, 0] * t[:, 1] - p[:, 1] * t[:, 0]
    delta_rad = torch.atan2(sin_d, cos_d)          
    delta_deg = torch.rad2deg(delta_rad)             

    mae_deg = delta_deg.abs().mean()
    mse_deg = (delta_deg ** 2).mean()

    pred_rad = torch.atan2(p[:, 1], p[:, 0])
    pred_deg = torch.rad2deg(pred_rad)
    true_rad = torch.atan2(t[:, 1], t[:, 0])
    true_deg = torch.rad2deg(true_rad)

    return mae_deg, mse_deg, pred_deg, true_deg


min_test_mae_seen = 1000000
min_test_mae_unseen = 1000000
norm_range_max = 132.0
norm_range_min = -132.0

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=2021, type=int)
parser.add_argument('--seed', default=32, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float)
parser.add_argument('--lr_regressor', default=5e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-10, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--model_path', default='../models/dino_resnet', type=str, help='path to dino_resnet pretrained model')
parser.add_argument('--train_image_dir', default='../pairUAV/train_tour', type=str, help='path to train images dir')
parser.add_argument('--train_json_dir', default='../pairUAV/train', type=str, help='path to train json dir')
parser.add_argument('--train_match_dir', default='./train_matches_data', type=str, help='path to train matches dir')
parser.add_argument('--test_image_dir', default='../pairUAV/test_tour', type=str, help='path to test images dir')
parser.add_argument('--test_json_dir', default='../pairUAV/test', type=str, help='path to test json dir')
parser.add_argument('--test_match_dir', default='./test_matches_data', type=str, help='path to test matches dir')
parser.add_argument('--test_output_txt', default='./test_predict_output.txt', type=str, help='path to write ordered test predictions')

class TourFrameDataset(Dataset):
    def __init__(self, image_dir, json_dir, match_dir, model_path='/root/dreamNav/models/dino_resnet', has_gt=True, force_image_ext=None):
        super().__init__()
        self.image_processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.image_dir = image_dir
        self.match_dir = match_dir
        self.has_gt = has_gt
        self.force_image_ext = force_image_ext
        self.sx, self.sy = 224.0 / 640.0, 224.0 / 480.0

        self.json_paths = []
        for name in os.listdir(json_dir):
            sub_path = os.path.join(json_dir, name)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith('.json'):
                        self.json_paths.append(os.path.join(sub_path, f))
            elif name.endswith('.json'):
                self.json_paths.append(sub_path)

        self.json_paths.sort(key=self._json_path_sort_key)

    @staticmethod
    def _extract_int(value):
        m = re.search(r'\d+', str(value))
        if m:
            return int(m.group())
        return float('inf')

    @classmethod
    def _json_path_sort_key(cls, json_path):
        p = Path(json_path)
        group_name = p.parent.name
        json_name = p.stem
        return (cls._extract_int(group_name), group_name, cls._extract_int(json_name), json_name)

    def _resolve_image_path(self, image_rel_path):
        candidates = []
        image_rel_path = str(image_rel_path)
        image_name = Path(image_rel_path).name
        image_stem = Path(image_name).stem

        candidates.append(os.path.join(self.image_dir, image_rel_path))
        candidates.append(os.path.join(self.image_dir, image_name))
        if self.force_image_ext:
            candidates.append(os.path.join(self.image_dir, image_stem + self.force_image_ext))

        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Image not found for {image_rel_path}, tried: {candidates}")

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, i):
        json_path = self.json_paths[i]
        p = Path(json_path)
        json_id = p.parent.name

        # 读 JSON
        with open(json_path, "rb") as f:
            data = json.load(f)

        a_path = self._resolve_image_path(data["image_a"])
        b_path = self._resolve_image_path(data["image_b"])
        a_name = Path(a_path).stem
        b_name = Path(b_path).stem

        if self.has_gt:
            npz_candidates = [
                os.path.join(self.match_dir, json_id, f"{a_name}_{b_name}_matches.npz"),
            ]
        else:
            npz_candidates = [
                os.path.join(self.match_dir, a_name, f"{b_name}.npz"),
                os.path.join(self.match_dir, a_name, f"{b_name}_matches.npz"),
                os.path.join(self.match_dir, a_name, f"{a_name}_{b_name}_matches.npz"),
            ]

        npz_path = None
        for c in npz_candidates:
            if os.path.exists(c):
                npz_path = c
                break
        if npz_path is None:
            raise FileNotFoundError(f"Match npz not found, tried: {npz_candidates}")

        with np.load(npz_path, allow_pickle=False) as z:
            k0 = z["keypoints0"]   # [N,2], (x,y)
            k1 = z["keypoints1"]   # [M,2], (x,y)
            m  = z["matches"]      # [N], 目标索引，<0 代表无效

        # 过滤无效匹配，向量化计算索引与位移
        valid = (m >= 0) & (m < len(k1))
        src = k0[valid]             # [K,2]
        dst = k1[m[valid]]          # [K,2]

        # 缩放到 224x224 网格
        x_src = src[:, 0] * self.sx
        y_src = src[:, 1] * self.sy
        x_dst = dst[:, 0] * self.sx
        y_dst = dst[:, 1] * self.sy

        # 注意 PyTorch 张量维度 [C,H,W]，索引应为 [y,x]
        xi = np.clip(x_src, 0, 223.9999).astype(np.int32)
        yi = np.clip(y_src, 0, 223.9999).astype(np.int32)

        dx = (x_dst - x_src).astype(np.float32)
        dy = (y_dst - y_src).astype(np.float32)

        # 一次性写入 (2,224,224)
        match_field = np.zeros((2, 224, 224), dtype=np.float32)
        match_field[0, yi, xi] = dx
        match_field[1, yi, xi] = dy
        match_tensor = torch.from_numpy(match_field)  # (2,224,224)

        # 加载图像
        with Image.open(a_path) as im:
            im = im.convert("RGB")
            img_t = self.image_processor(images=im, return_tensors="pt").pixel_values.squeeze(0)

        # 拼接
        final_input = torch.cat((img_t, match_tensor), dim=0)  # (3+2,224,224)

        if not self.has_gt:
            return final_input, json_path

        theta_deg = float(data["heading_num"])
        theta_rad = math.radians(theta_deg)

        label_vec = torch.tensor([math.cos(theta_rad), math.sin(theta_rad)], dtype=torch.float32)
        label_deg = torch.tensor(theta_deg, dtype=torch.float32)

        range_num = float(data['range_num'])
        norm_range = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

        return final_input, label_vec, label_deg, json_path, range_num, norm_range

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoFeatureExtractor, ResNetModel
from PIL import Image
import requests

class OurModel(nn.Module):
    def __init__(self, model_path='/root/dreamNav/models/dino_resnet', pretrained=True):
        super(OurModel, self).__init__()

        original_resnet = ResNetModel.from_pretrained(model_path)
        #for param in original_resnet.parameters():
        #    param.requires_grad = False

        self.input_conv = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.model = nn.Sequential(*list(original_resnet.children())[:-1]) 
        self.regressor1 = nn.Linear(2048, 128)
        self.regressor2 = nn.Linear(128, 3)  

    def forward(self, images_a):
        images_a = self.input_conv(images_a)
        hidden_states_a = self.model(images_a).last_hidden_state
        pooled_output = torch.mean(hidden_states_a, dim=[2, 3]) 
        output = self.regressor2(self.regressor1(pooled_output))
        return output 


def main():

    open("output.log", "w").close()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading model...', flush=True)
    model = OurModel(model_path=args.model_path)
    model.to(device)
    model = torch.nn.DataParallel(model)
    print(f'Model loaded, using {torch.cuda.device_count()} GPUs', flush=True)
    
    # 打印模型信息
    print_model_info(model, device)

    print('Loading datasets...', flush=True)
    train_dataset = TourFrameDataset(
        args.train_image_dir,
        args.train_json_dir,
        args.train_match_dir,
        model_path=args.model_path,
        has_gt=True,
        force_image_ext=None,
    )
    test_dataset = TourFrameDataset(
        args.test_image_dir,
        args.test_json_dir,
        args.test_match_dir,
        model_path=args.model_path,
        has_gt=False,
        force_image_ext='.webp',
    )
    print('Datasets loaded', flush=True)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=2)

    optimizer = torch.optim.SGD([
        {'params': model.module.input_conv.parameters(), 'lr': args.lr_regressor},
        {'params': model.module.model.parameters(), 'lr': args.lr},
        {'params': list(model.module.regressor1.parameters()) + list(model.module.regressor2.parameters()), 'lr': args.lr_regressor}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        step_size=30,  
        gamma=0.1
    )
    
    for epoch in range(args.epochs):
        train(train_dataloader, model, optimizer, epoch, device, args)
        
        scheduler.step()

    run_test_and_save_txt(test_dataloader, model, device, args.test_output_txt)
    
    
def train(train_loader, model, optimizer, epoch, device, args):
    
    use_accel = True
    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses_dir = AverageMeter('Loss_dir', use_accel, ':.4e', Summary.NONE)  
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_dir],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    epoch_start_time = time.time()

    for i, (image_a, label_vec, label_deg, _, label_range, label_norm_range) in enumerate(train_loader):
        model.train()
        data_time.update(time.time() - end)

        image_a = image_a.to(device, non_blocking=True)
        label_vec = label_vec.to(device, non_blocking=True)  
        label_range = label_range.to(device, non_blocking=True) 
        label_norm_range = label_norm_range.to(device, non_blocking=True) 

        output = model(image_a)

        angle_lossnum = angle_loss_cos_sin(output[:, :2], label_vec)
        range_lossnum = range_loss(output[:, 2], label_norm_range) 
        loss = angle_lossnum + range_lossnum

        #print(angle_lossnum, range_lossnum)
        losses_dir.update(loss.item(), image_a.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            
            #test_angle_mse, test_angle_mae, test_range_mse, test_range_mae, test_success_rate = validate(test_dataloader, model, criterion, args, device, quick=True, tag="seen")
            #print(f'Seen: ANGLE: mse:{test_angle_mse:.2f} | mae:{test_angle_mae:.2f} RANGE: mse: {test_range_mse:.2f} | mae: {test_range_mae:.2f} SR: {test_success_rate*100:.2f}%')
        

    epoch_time = time.time() - epoch_start_time
    print(f'\n===== Epoch {epoch} Finished (Time: {epoch_time:.2f}s) =====')
    with open('output.log', 'a') as outf:
        outf.write(f'\n===== Epoch {epoch} Summary =====\n')
        outf.write(f'Train epoch finished in {epoch_time:.2f}s\n')
        outf.write(f'=====================================\n\n')
        outf.flush()
    
    return 

def extract_group_json_sort_key(json_path):
    p = Path(json_path)
    group_value = p.parent.name
    json_value = p.stem
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        group_value = data.get('group_id', group_value)
        json_value = data.get('json_id', json_value)
    except Exception:
        pass

    group_num = TourFrameDataset._extract_int(group_value)
    json_num = TourFrameDataset._extract_int(json_value)
    return (group_num, str(group_value), json_num, str(json_value))

def run_test_and_save_txt(test_loader, model, device, output_txt_path):
    print('Running final test inference...', flush=True)
    model.eval()
    results = []

    with torch.no_grad():
        for image_a, json_paths in tqdm(test_loader, desc='Test Inference', leave=False):
            image_a = image_a.to(device, non_blocking=True)
            output = model(image_a)

            pred_xy = F.normalize(output[:, :2], dim=-1)
            pred_rad = torch.atan2(pred_xy[:, 1], pred_xy[:, 0])
            pred_deg = torch.rad2deg(pred_rad)
            pred_range = output[:, 2] * (norm_range_max - norm_range_min) + norm_range_min

            pred_deg_list = pred_deg.detach().cpu().tolist()
            pred_range_list = pred_range.detach().cpu().tolist()

            for jp, deg, rag in zip(json_paths, pred_deg_list, pred_range_list):
                results.append((jp, float(deg), float(rag)))

    results.sort(key=lambda x: extract_group_json_sort_key(x[0]))

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for _, heading_num, range_num in results:
            f.write(f'{heading_num:.6f} {range_num:.6f}\n')

    print(f'Final test output written to: {output_txt_path}', flush=True)

def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)  
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def validate(val_loader, model, criterion, args, device, quick=True, tag='seen'):

    global min_test_mae_seen
    global min_test_mae_unseen

    use_accel = True
    now_deg_preds = []
    now_deg_trues = []
    now_rag_preds = []
    now_rag_trues = []

    now_jsons = []
    
    success_count = 0
    total_count = 0

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses_mse_deg = AverageMeter('MSE_deg', use_accel, ':.4e', Summary.NONE)
    losses_mae_deg = AverageMeter('MAE_deg', use_accel, ':.4e', Summary.NONE)
    losses_mse_rag = AverageMeter('MSE_rag', use_accel, ':.4e', Summary.NONE)
    losses_mae_rag = AverageMeter('MAE_rag', use_accel, ':.4e', Summary.NONE)
    losses_dir = AverageMeter('Loss_dir', use_accel, ':.4e', Summary.NONE)  
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_dir, losses_mse_deg, losses_mae_deg, losses_mse_rag, losses_mae_rag],
        prefix='Test: ')

    model.eval()
    total_samples = 0
    inference_start_time = time.time()
    
    with torch.no_grad():
        end = time.time()
        for i, (image_a, label_vec, label_deg, json_paths, label_range, label_norm_range) in enumerate(val_loader):
            image_a = image_a.to(device, non_blocking=True)
            label_vec = label_vec.to(device, non_blocking=True) 
            label_deg = label_deg.to(device, non_blocking=True) 
            label_range = label_range.to(device, non_blocking=True) 
            label_norm_range = label_norm_range.to(device, non_blocking=True) 

            output = model(image_a)  
            total_samples += image_a.size(0)

            loss_dir = angle_loss_cos_sin(output[:, :2], label_vec)
            mae_deg, mse_deg, pred_deg, true_deg = wrapped_angle_error_deg(output[:, :2], label_vec)

            pred_range = output[:, 2] * (norm_range_max - norm_range_min) + norm_range_min
            mae_rag, mse_rag = wrapped_range_error(pred_range, label_range)

            # 计算成功率：预测终点与真实终点距离 < 10m
            pred_rad = torch.atan2(output[:, 1], output[:, 0])
            true_rad = torch.atan2(label_vec[:, 1], label_vec[:, 0])
            pred_x = pred_range * torch.cos(pred_rad)
            pred_y = pred_range * torch.sin(pred_rad)
            true_x = label_range * torch.cos(true_rad)
            true_y = label_range * torch.sin(true_rad)
            endpoint_dist = torch.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            success_count += (endpoint_dist < 10.0).sum().item()
            total_count += image_a.size(0)

            losses_dir.update(loss_dir.item(), image_a.size(0))
            losses_mse_deg.update(mse_deg.item(), image_a.size(0))
            losses_mae_deg.update(mae_deg.item(), image_a.size(0))
            losses_mse_rag.update(mse_rag, image_a.size(0))
            losses_mae_rag.update(mae_rag, image_a.size(0))

            now_deg_preds.extend(pred_deg.detach().cpu().tolist())
            now_deg_trues.extend(true_deg.detach().cpu().tolist())
            now_rag_preds.extend(pred_range.detach().cpu().tolist())
            now_rag_trues.extend(label_range.detach().cpu().tolist())

            now_jsons.extend(list(json_paths))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)
                if quick:
                    break
    
    # 计算推理速度
    inference_time = time.time() - inference_start_time
    throughput = total_samples / inference_time if inference_time > 0 else 0
    print(f"Inference Throughput ({tag}): {throughput:.2f} samples/s")
    
    success_rate = success_count / total_count if total_count > 0 else 0.0
    
    if quick:
        return losses_mse_deg.avg, losses_mae_deg.avg, losses_mse_rag.avg, losses_mae_rag.avg, success_rate
    
    now_loss_all = losses_mae_deg.avg + losses_mae_rag.avg

    have_update = False
    if tag == "seen":
        if now_loss_all < min_test_mae_seen:
            have_update = True
            min_test_mae_seen = now_loss_all
    elif tag == "unseen":
        if now_loss_all < min_test_mae_unseen:
            have_update = True
            min_test_mae_unseen = now_loss_all
    
    if have_update:
        pred_deg_num = now_deg_preds
        true_deg_num = now_deg_trues
        pred_rag_num = now_rag_preds
        true_rag_num = now_rag_trues
        json_num = now_jsons

        now_json = {'pred_deg_num': pred_deg_num, 'true_deg_num': true_deg_num, 'pred_rag_num': pred_rag_num, 'true_rag_num': true_rag_num, 'json_path': json_num}
        with open('step1_' + tag + '.json', 'w') as f:
            json.dump(now_json, f)
        
    return losses_mse_deg.avg, losses_mae_deg.avg, losses_mse_rag.avg, losses_mae_rag.avg, success_rate

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()