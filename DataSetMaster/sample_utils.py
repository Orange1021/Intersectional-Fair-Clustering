import torch
from torch.utils import data
import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.optimize import linprog
import warnings
from numba import NumbaDeprecationWarning
import random

# 忽略Numba的弃用警告
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

class SamplingDataset(data.Dataset):
    def __init__(self, dataset):
        """
        通用采样数据集类
        :param dataset: 原始数据集，需要返回 (img, g_list, target, idx) 格式的数据
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.sampled_indices = None
        
        # 加载数据
        self._load_data()
        
        # 初始化采样相关变量
        self.samples_check = torch.zeros(len(self.targets))
        self.targets_bin = self.get_targets_bin()
        
        # 设置DRO信息
        self.set_dro_info()
    
    def _load_data(self):
        """加载原始数据集的数据"""
        self.samples = []
        self.targets = []
        self.bias_targets = []
        
        for idx in self.indices:
            img, g_list, target, _ = self.dataset[idx]
            self.samples.append(img)
            self.targets.append(target)
            
            # 处理敏感属性
            if isinstance(g_list, list):
                # 如果有多个敏感属性，将它们组合成一个偏置标签
                # g_list[0]是奇偶性(0/1)，g_list[1]是数据源(0/1)
                # 使用公式：bias = 奇偶性 + 2 * 数据源
                if isinstance(g_list[0], torch.Tensor):
                    g0 = g_list[0].item()
                else:
                    g0 = g_list[0]
                    
                if isinstance(g_list[1], torch.Tensor):
                    g1 = g_list[1].item()
                else:
                    g1 = g_list[1]
                    
                bias = g0 + 2 * g1  # 这样会得到0,1,2,3四种组合
                self.bias_targets.append(bias)
            else:
                # 如果只有一个敏感属性
                if isinstance(g_list, torch.Tensor):
                    g_list = g_list.item()
                self.bias_targets.append(g_list)
        
        # 转换为tensor
        self.targets = torch.tensor(self.targets)
        self.bias_targets = torch.tensor(self.bias_targets)
    
    def get_targets_bin(self):
        """获取二值化的目标标签"""
        return self.targets
    
    def set_dro_info(self):
        """设置DRO（Distributionally Robust Optimization）信息"""
        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets))
        
        # 为每个样本分配组别索引
        self.groups_idx = torch.zeros((len(self.targets)))
        for i, t, b in zip(torch.arange(len(self.targets)), self.targets, self.bias_targets):
            idx = t + (b * num_targets)
            self.groups_idx[i] = idx
        
        self.n_groups = num_targets * num_biases
    
    def group_counts(self):
        """计算每个组的样本数量"""
        counts = torch.zeros(self.n_groups)
        for i in range(self.n_groups):
            counts[i] = torch.sum(self.groups_idx == i)
        return counts
    
    def get_target_distro(self, target):
        """获取指定目标类别的偏置分布"""
        num_biases = len(torch.unique(self.bias_targets))
        target_distro = []
        for bias in range(num_biases):
            target_distro.append(torch.sum(torch.logical_and(self.targets == target,
                                                           self.bias_targets == bias)))
        return target_distro

    def solve_linear_program(self, target_distro, target_prime_distro):
        """使用线性规划求解采样比例"""
        num_biases = len(torch.unique(self.bias_targets))
        obj = [-1] * num_biases
        
        lhs_ineq = []
        for bias in range(num_biases):
            ineq = [0] * num_biases
            ineq[bias] = 1
            lhs_ineq.append(ineq)
        
        rhs_ineq = target_prime_distro

        lhs_eq = []
        target_distro = [x/sum(target_distro) for x in target_distro]
        for prob, bias in zip(target_distro, range(num_biases - 1)):
            eq = [-prob]*num_biases
            eq[bias] = 1 - prob
            lhs_eq.append(eq)
        
        rhs_eq = [0]*(num_biases - 1)
        bnd = [(0, float("inf")) for _ in range(num_biases)]

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                     A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                     method="revised simplex")
        
        sol = opt.x
        sol = [int(x) for x in sol]
        sol = [x if x > 0 else 1 for x in sol]
        return sol

    def get_kept_indices(self, target, target_prime, target_prime_new_distro):
        """获取需要保留的样本索引"""
        to_keep_indices = []
        for bias, bias_distro in enumerate(target_prime_new_distro):
            tmp = torch.logical_and(self.targets == target_prime, self.bias_targets == bias)
            indices_bias = list(torch.arange(len(self.targets))[tmp].numpy())
            to_keep_indices.extend(np.random.choice(indices_bias, bias_distro, replace=False))
        return to_keep_indices

    def bias_mimick(self):
        """应用偏置模仿采样"""
        print("\n偏置模仿采样信息:")
        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets))
        
        # 初始化二值化目标标签
        self.targets_bin = torch.zeros((len(self.targets), num_targets))
        self.targets_bin[torch.arange(len(self.targets)), self.targets] = 1
        
        # 对每个目标类别进行处理
        for target in range(num_targets):
            print(f"\n处理目标类别 {target}:")
            target_distro = self.get_target_distro(target)
            print(f"目标类别 {target} 的原始分布: {[x.item() for x in target_distro]}")
            
            to_keep_indices = []
            for target_prime in range(num_targets):
                if target_prime == target:
                    # 对于相同目标类别，保留所有样本
                    indices_target = list(torch.arange(len(self.targets))[self.targets == target])
                    to_keep_indices.extend(indices_target)
                   
                else:
                    # 对于不同目标类别，使用线性规划确定采样数量
                    target_prime_distro = self.get_target_distro(target_prime)
                    target_prime_new_distro = self.solve_linear_program(target_distro, target_prime_distro)
                    kept = self.get_kept_indices(target, target_prime, target_prime_new_distro)
                    to_keep_indices.extend(kept)
                 
            
            # 标记未被保留的样本
            full_idxs = torch.arange(len(self.targets))
            to_select = torch.ones(len(self.targets))
            to_select[to_keep_indices] = 0
            full_idxs = full_idxs[to_select.bool()]
            self.targets_bin[full_idxs, target] = -1
        
        # 根据 targets_bin 确定最终要保留的样本
        valid_samples = torch.all(self.targets_bin != -1, dim=1)
        self.sampled_indices = list(torch.arange(len(self.targets))[valid_samples].numpy())
        
        print(f"\n采样后总样本数: {len(self.sampled_indices)}")
        print("采样后的分布:")
        self.print_new_distro()
    
    def under_sample_ce(self):
        """应用欠采样"""
        print("\n欠采样信息:")
        group_counts = self.group_counts()
        min_count = int(torch.min(group_counts).item())
        print(f"最小组样本数: {min_count}")
        
        to_keep_idx = []
        for group_idx in range(len(group_counts)):
            if group_counts[group_idx] == 0:
                print(f"跳过空组 {group_idx}")
                continue
                
            indices = torch.logical_and(self.groups_idx == group_idx, self.samples_check == 0)
            print(f"组 {group_idx} 的初始可用样本数: {torch.sum(indices)}")
            
            if torch.sum(indices) < min_count:
                # 如果可用样本不足，重置该组的采样状态
                self.samples_check[self.groups_idx == group_idx] = 0
                indices = torch.logical_and(self.groups_idx == group_idx, self.samples_check == 0)
                print(f"组 {group_idx} 重置后的可用样本数: {torch.sum(indices)}")
            
            subsampled_indices = torch.arange(len(self.targets))[indices]
            subsampled_indices = list(subsampled_indices.numpy())
            
            # 使用 random.sample 进行无放回采样
            subsampled_indices = random.sample(subsampled_indices, min_count)
            self.samples_check[subsampled_indices] = 1
            print(f"组 {group_idx} 采样后的样本数: {len(subsampled_indices)}")
            
            to_keep_idx.extend(subsampled_indices)
        
        print(f"\n采样后总样本数: {len(to_keep_idx)}")
        self.sampled_indices = to_keep_idx
    
    def over_sample_ce(self):
        """应用过采样"""
        print("\n过采样信息:")
        group_counts = self.group_counts()
        max_count = int(torch.max(group_counts).item())
        print(f"最大组样本数: {max_count}")
        
        # 设置最大采样倍数（1.5倍）
        max_multiplier = 1.5
        
        to_keep_idx = []
        for group_idx in range(len(group_counts)):
            if group_counts[group_idx] == 0:
                print(f"跳过空组 {group_idx}")
                continue
                
            original_count = int(group_counts[group_idx].item())
            print(f"组 {group_idx} 的初始样本数: {original_count}")
            
            # 计算目标采样数量，不超过原始样本数的1.5倍
            target_count = min(max_count, int(original_count * max_multiplier))
            
            subsampled_indices = torch.arange(len(self.targets))[self.groups_idx == group_idx]
            subsampled_indices = list(subsampled_indices.numpy())
            
            # 使用 random.choices 进行有放回采样
            subsampled_indices = random.choices(subsampled_indices, k=target_count)
            print(f"组 {group_idx} 采样后的样本数: {len(subsampled_indices)}")
            
            to_keep_idx.extend(subsampled_indices)
        
        print(f"\n采样后总样本数: {len(to_keep_idx)}")
        self.sampled_indices = to_keep_idx
    
    def __getitem__(self, idx):
        if self.sampled_indices is not None:
            idx = self.sampled_indices[idx]
        return self.dataset[idx]
    
    def __len__(self):
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return len(self.dataset)
    
    def print_new_distro(self):
        """打印数据分布"""
        if self.sampled_indices is None:
            # 显示原始数据分布
            print("\n原始数据分布:")
            print("-" * 50)
            group_counts = {}
            for idx in range(len(self.dataset)):
                _, g_list, _, _ = self.dataset[idx]
                if isinstance(g_list, list):
                    # 处理列表中的每个元素，可能是张量或整数
                    key = tuple(g.item() if isinstance(g, torch.Tensor) else g for g in g_list)
                else:
                    # 处理单个值，可能是张量或整数
                    key = (g_list.item() if isinstance(g_list, torch.Tensor) else g_list,)
                group_counts[key] = group_counts.get(key, 0) + 1
            
            print("敏感属性分布:")
            total = sum(group_counts.values())
            for groups, count in sorted(group_counts.items()):
                print(f"组 {groups}: {count} 个样本 ({count/total*100:.2f}%)")
            return
            
        # 显示采样后的分布
        print("\n采样后的敏感属性分布:")
        group_counts = {}
        for idx in self.sampled_indices:
            _, g_list, _, _ = self.dataset[idx]
            if isinstance(g_list, list):
                # 处理列表中的每个元素，可能是张量或整数
                key = tuple(g.item() if isinstance(g, torch.Tensor) else g for g in g_list)
            else:
                # 处理单个值，可能是张量或整数
                key = (g_list.item() if isinstance(g_list, torch.Tensor) else g_list,)
            group_counts[key] = group_counts.get(key, 0) + 1
            
        total = sum(group_counts.values())
        for groups, count in sorted(group_counts.items()):
            print(f"组 {groups}: {count} 个样本 ({count/total*100:.2f}%)")

class MultiSensitiveSamplingDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        attr_indices = []
        num_attrs = len(dataset[0][1]) if isinstance(dataset[0][1], list) else 1
        for attr_idx in range(num_attrs):
            # 统计该属性所有取值
            value_to_indices = {}
            for i in self.indices:
                val = dataset[i][1][attr_idx]
                value_to_indices.setdefault(val, []).append(i)
            # 找到最小组
            min_count = min(len(idxs) for idxs in value_to_indices.values())
            if min_count == 0:
                continue
            # 每个取值采样min_count个
            sampled = []
            for idxs in value_to_indices.values():
                if len(idxs) >= min_count:
                    sampled.extend(random.sample(idxs, min_count))
            attr_indices.append(set(sampled))
        # 取所有属性采样结果的并集
        if attr_indices:
            self.sampled_indices = list(set.union(*attr_indices))
        else:
            self.sampled_indices = self.indices  # fallback: 全部
    def __getitem__(self, idx):
        real_idx = self.sampled_indices[idx]
        return self.dataset[real_idx]
    def __len__(self):
        return len(self.sampled_indices)

def apply_sampling(dataset: data.Dataset, method: str = 'bm') -> SamplingDataset:
    """
    应用采样方法到数据集
    :param dataset: 原始数据集
    :param method: 采样方法 ('bm', 'us', 'os')
    :return: 采样后的数据集
    """
    sampling_dataset = SamplingDataset(dataset)
    
    if method == 'bm':
        sampling_dataset.bias_mimick()
    elif method == 'us':
        sampling_dataset.under_sample_ce()
    elif method == 'os':
        sampling_dataset.over_sample_ce()
    else:
        raise ValueError(f"不支持的采样方法: {method}")
    
    return sampling_dataset

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    import sys
    import os
    import argparse
    
    # 添加父目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from DataSetMaster import dataset
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建参数对象
    class Args:
        def __init__(self):
            self.dataset = 'MNISTUSPS'  # 改为 MNISTUSPS
            self.FeatureType = 'Default'
            self.MnistTrain = False
    
    args = Args()
    
    # 创建数据转换
    transform = transforms.Compose([
        transforms.Resize(28),  # MNISTUSPS 使用 28x28 的图像大小
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # MNISTUSPS 使用单通道图像
    ])
    
    # 获取 MNISTUSPS 数据集
    print("加载 MNISTUSPS 数据集...")
    train_loader, test_loader, class_num = dataset.get_dataloader(
        dataset='MNISTUSPS',
        batch_size=512,
        sampling_method=None,  # 先不进行采样
        args=args
    )
    
    # 获取原始数据集
    original_dataset = train_loader.dataset
    
    # 打印原始数据分布
    print("\n原始数据分布:")
    print("-" * 50)
    original_sampling = SamplingDataset(original_dataset)
    original_sampling.print_new_distro()
    
    # 测试不同的采样方法
    sampling_methods = ['bm', 'us', 'os']
    
    for method in sampling_methods:
        print(f"\n测试 {method} 采样方法:")
        print("-" * 50)
        
        # 应用采样
        sampled_dataset = apply_sampling(original_dataset, method)
        
        # 打印采样后的分布
        print(f"\n采样方法 {method} 的结果:")
        sampled_dataset.print_new_distro()
        
        # 创建数据加载器并测试几个批次
        sampled_loader = torch.utils.data.DataLoader(
            sampled_dataset,
            batch_size=32,
            shuffle=True
        )
        
        # 测试几个批次
        print("\n测试数据加载:")
        for i, (imgs, g_list, targets, indices) in enumerate(sampled_loader):
            print(f"批次 {i+1}:")
            print(f"  图像形状: {imgs.shape}")
            print(f"  目标形状: {targets.shape}")
            print(f"  敏感属性形状: {[g.shape for g in g_list]}")
            print(f"  索引形状: {indices.shape}")
            if i >= 2:  # 只测试前3个批次
                break
        
        print("\n" + "="*50)
