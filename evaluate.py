import os
import time

import torch
# from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from munkres import Munkres
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc

from Experiment.Visualize import PrintTimer
from Utils.DirectoryOperator import FileOperator

import torch.nn.functional as F
from npeet import entropy_estimators as ee

# 添加全局变量记录最佳结果
best_balance = 0.0
best_nmi = 0.0
best_ari = 0.0
best_acc = 0.0

BestMnce = 0.0  # 添加这行
BestCGF = None
BestDeltaKNN = 0.0

def evaluate(feature_vec, pred_vec, type_vec, group_vec, best_acc, fair_metric=False):
    # tqdm.write("Evaluating the clustering results...")
    # print('Evaluating the clustering results... ')
    nmi, ari, acc, pred_adjusted = cluster_metrics(type_vec, pred_vec)
    if best_acc < acc:
        best_acc = acc
    print('nmi={:5.02f}, ari={:5.02f}, acc={:5.02f}, BestAcc={:5.02f}'.format(
        nmi * 100, ari * 100, acc * 100, best_acc * 100))
    # tqdm.write('NMI=%.4f, ACC=%.4f, ARI=%.4f' % (nmi, acc, ari), end='')
    if fair_metric:
        kl, ari_b = fair_metrics(feature_vec, group_vec, pred_vec, type_vec)
        print(', KL=%.4f, ARI_b=%.4f' % (kl, ari_b), end='')
    # tqdm.write('')
    return best_acc


BestAcc = 0.
BestAri = 0.
BestNmi = 0.
BestBalance = 0.
BestFairness = 0.
BestNmiFair_avg = 0.
BestNmiFair = 0.
BestFmeasure = 0.
BestEntropy = 0.


def calculate_entropy(input):
    """
    calculates the entropy score
    :param input: tensor for which entropy needs to be calculated
    """
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def calculate_balance(predicted, size_0, k=10):
    """
    calculates the balance score of a model output_0
    :param predicted: tensor with the model predictions (n_samples_g0+n_samples_g1, )V(0:cluster)
    :param size_0: size of sample
    :param k: amount of clusters
    """
    count = torch.zeros((k, 2))
    # cluster, grounp
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5
    print(count)
    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])
    # print(balance_1)
    en_0 = calculate_entropy(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = calculate_entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()


def my_balance(predicted, g, cluster_num, group_num):
    """

    :param predicted:
    :param g:
    :param cluster_num:
    :param group_num:
    :return: balance, entro: entropy of the proportion of the size of the clusters of i-th group.
    """
    count = torch.zeros((cluster_num, group_num))
    # cluster, grounp
    # count_time = 0
    for ci, gi in zip(predicted, g):
        # count_time += 1
        # print(ci, gi)
        # print(ci)
        # print(gi)
        count[ci, gi] += 1
        # print(count_time, count)
    # print(count_time)
    # print(predicted.shape)
    # print(g.shape)
    # print(count)

    count[count == 0] = 1e-5
    balance_v = torch.amin(
        torch.amin(count, dim=1) / torch.amax(count, dim=1)
    )
    # print(balance_v)

    epsilon = 1e-5
    prob = count / torch.sum(count, dim=0, keepdim=True)
    entro = torch.sum(-prob * torch.log(prob + epsilon), dim=0)

    # print(entro)

    return balance_v.numpy(), entro.numpy()


def my_balance_intersectional(predicted, group_vec_list, class_num):
    """
    计算交叉属性的平衡性分数（my_balance的"木桶效应"逻辑扩展版）
    :param predicted: 预测的聚类标签 (Tensor or Numpy Array)
    :param group_vec_list: 敏感属性的列表，每个元素是一个属性向量 (list of Tensors or Numpy Arrays)
    :param class_num: 聚类的数量
    :return: 交叉平衡性分数
    """
    if not isinstance(predicted, np.ndarray):
        predicted = predicted.cpu().numpy()

    # 1. 创建交叉属性
    # 将group_vec_list中的张量或数组转换为numpy数组
    groups_np = [g.cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g) for g in group_vec_list]
    
    # 将多个属性向量堆叠成一个 (n_samples, n_attributes) 的数组
    cross_group_tuples = list(zip(*groups_np))
    
    # 使用pandas.factorize为每个独特的元组（交叉组）分配一个整数ID
    # `intersectional_g`是每个样本所属交叉组的ID, `unique_cross_groups`是所有独特的交叉组
    intersectional_g, unique_cross_groups = pd.factorize(cross_group_tuples)
    group_num = len(unique_cross_groups)
    
    print(f"检测到 {len(group_vec_list)} 个敏感属性，组合成 {group_num} 个交叉群组。")

    # 2. 计算平衡性
    # 确保数据类型正确
    predicted = torch.as_tensor(predicted, dtype=torch.long)
    intersectional_g = torch.as_tensor(intersectional_g, dtype=torch.long)

    # 创建一个 (聚类数 x 交叉组数) 的计数矩阵
    count = torch.zeros((class_num, group_num))
    for ci, gi in zip(predicted, intersectional_g):
        count[ci, gi] += 1
    
    # 避免除以零
    count[count == 0] = 1e-5  # 使用一个很小的数代替0，避免后续计算问题
    
    # 3. 计算 min/max 比率
    # a. 对每个聚类（每一行），计算各组数量的min/max比率
    # amax(dim=1) 找到每行（每个聚类）中样本数最多的那个交叉组的数量
    # amin(dim=1) 找到每行（每个聚类）中样本数最少的那个交叉组的数量
    balance_per_cluster = torch.amin(count, dim=1) / torch.amax(count, dim=1)

    # b. 在所有聚类的平衡性分数中，取最小值（木桶效应）
    final_balance_score = torch.amin(balance_per_cluster)

    return final_balance_score.numpy()


# if __name__ == '__main__':
#     print('[{}]'.format(1))
# predicted = torch.arange(100) % 3
# g =torch.as_tensor( torch.arange(100) > 50, dtype=torch.int)
# print(predicted.shape)
# my_balance(predicted.cuda(), g.cuda(), cluster_num=3, group_num=2)
# print(predicted.shape)
# b, e0, e1 = balance(predicted=predicted, size_0=(g==0).sum(), k=3 )
# print(b, e0, e1)
def normalized_mutual_information_without_mean(o):
    """

    :param o: o_cg = p_cg
    :return: 1-NMI(c,g)
    """

    def entroph(v):
        """

        :param v: element-wise entroph
        :return:
        """
        return -torch.sum(torch.log(v) * v)

    hc = entroph(torch.sum(o, dim=1))
    hg = entroph(torch.sum(o, dim=0))
    hclg = -torch.sum(torch.log(o / torch.sum(o, dim=0, keepdim=True)) * o)
    icg = hc - hclg
    # nmi = icg / ((hc + hg) / 2)
    nmi = icg / hc
    # print('o == {}'.format(o))
    # print('icg == {}'.format(icg))
    # print('hclg == {}'.format(hclg))
    # print('hc == {}'.format(hc))
    # print('hg == {}'.format(hg))
    # print('nmi == {}'.format(nmi))
    return 1 - nmi


def normalized_mutual_information(o):
    """

    :param o: o_cg = p_cg
    :return: 1-NMI(c,g)
    """

    def entroph(v):
        """

        :param v: vector or matrix(trait to be vector)
        :return:element-wise entroph
        """
        # print(v)
        # print(torch.log(v))
        # print(torch.log(v) * v)
        # print(-torch.sum(torch.log(v) * v))
        # vv = v[v!=0]
        return -torch.sum(torch.log((v + 1e-10 * (v == 0))) * v)

    # hc = entroph(torch.sum(o, dim=1))
    hg = entroph(torch.sum(o, dim=0))
    # pg = torch.sum(o, dim=0, keepdim=True)
    # pclg = o / (pg + 1e-10 * (pg == 0))
    # hclg = -torch.sum(torch.log(pclg + 1e-10 * (pclg == 0)) * o)

    pc = torch.sum(o, dim=1, keepdim=True)
    pglc = o / (pc + 1e-10 * (pc == 0))
    pglc[torch.sum(pglc, dim=1, keepdim=False) == 0] = 1 / len(pglc[0])
    hglc_c = -torch.sum(torch.log(pglc + 1e-10 * (pglc == 0)) * pglc, dim=1)
    # hglc = torch.sum(o,  dim=1) @ hglc_c
    # icg = hc - hclg
    # icg = hg - hglc
    # nmi = icg / ((hc + hg) / 2)
    # hclx = 0.088141
    # icx = hc - hclx
    # nmi = icg / hg
    # print('o == {}'.format(o))
    # print('icg == {}'.format(icg))
    # print('hclg == {}'.format(hclg))
    # print('hglc == {}'.format(hglc))
    # print('hc == {}'.format(hc))
    # print('hg == {}'.format(hg))
    # print('nmi == {}'.format(nmi))
    # print()
    # return 1-nmi
    return torch.min(hglc_c) / hg


class FMeasure:
    def __init__(self, beta=1):
        """

        :param beta: r/p = beta, s.t. dr = dp
        """
        self.beta = beta
    def __call__(self, p, r):
        return (self.beta ** 2 + 1) * (p * r) / (self.beta ** 2 * p + r)


def relative_fairness(O, E):
    return 1 - torch.sum(O * torch.log(O / E))


def multi_balance(predicted, g, class_num, group_num):
    """
    计算多敏感属性的平衡性分数 - 改进版
    :param predicted: 预测的聚类标签
    :param g: 敏感属性列表，每个元素对应一个敏感属性
    :param class_num: 聚类数量
    :param group_num: 每个敏感属性的组数
    :return: 平衡性分数列表和详细统计信息
    """
    balance_scores = []
    balance_details = []
    
    # 对每个敏感属性分别计算平衡性分数
    for group_vec in g:
        # 确保预测标签从0开始
        min_label = np.min(predicted)
        predicted_normalized = predicted - min_label
        
        # 确保class_num正确
        actual_class_num = len(np.unique(predicted_normalized))
        if class_num != actual_class_num:
            print(f"警告: 提供的class_num={class_num}与实际聚类数{actual_class_num}不匹配，使用实际值")
            class_num = actual_class_num
        
        # 计算每个聚类中各个组的数量
        cluster_group = np.zeros((class_num, group_num))
        
        # 调试信息
        print(f"调试: predicted_normalized范围: {np.min(predicted_normalized)}到{np.max(predicted_normalized)}")
        print(f"调试: group_vec范围: {np.min(group_vec)}到{np.max(group_vec)}")
        print(f"调试: cluster_group形状: {cluster_group.shape}")
        
        for ci, gi in zip(predicted_normalized, group_vec):
            # 确保索引在有效范围内
            if ci >= class_num:
                print(f"警告: 聚类索引{ci}超出了预期范围{class_num}，跳过此样本")
                continue
            if gi >= group_num:
                print(f"警告: 组索引{gi}超出了预期范围{group_num}，跳过此样本")
                continue
            
            cluster_group[ci, gi] += 1
        
        # 记录详细统计信息
        detail = {"cluster_distribution": cluster_group.tolist()}
        
        # 计算每个聚类内的组平衡性（最小组比例/最大组比例）
        cluster_balance_scores = []
        for cluster_idx in range(class_num):
            cluster_counts = cluster_group[cluster_idx]
            if np.sum(cluster_counts) > 0:  # 避免空聚类
                # 计算组比例
                group_ratios = cluster_counts / np.sum(cluster_counts)
                # 找出非零比例
                nonzero_ratios = group_ratios[group_ratios > 0]
                if len(nonzero_ratios) > 1:  # 至少有两个组
                    # 计算最小比例与最大比例之比
                    balance = np.min(nonzero_ratios) / np.max(nonzero_ratios)
                    cluster_balance_scores.append(balance)
                else:
                    # 只有一个组，完全不平衡
                    cluster_balance_scores.append(0.0)
            else:
                # 空聚类视为完全不平衡
                cluster_balance_scores.append(0.0)
        
        # 总体平衡性分数是所有非空聚类的平均平衡性
        valid_scores = [s for s in cluster_balance_scores if s > 0]
        if valid_scores:
            avg_balance = np.mean(valid_scores)
        else:
            avg_balance = 0.0
        
        # 记录详细信息
        detail["cluster_balance_scores"] = cluster_balance_scores
        detail["overall_balance"] = avg_balance
        
        balance_scores.append(avg_balance)
        balance_details.append(detail)
    
    return balance_scores, balance_details


def multi_normalized_mutual_information(o_list):
    """
    计算多个敏感属性的归一化互信息
    :param o_list: 每个敏感属性的混淆矩阵列表
    :return: 每个敏感属性的NMI值
    """
    nmi_scores = []
    for o in o_list:
        hg = entroph(torch.sum(o, dim=0))
        pc = torch.sum(o, dim=1, keepdim=True)
        pglc = o / (pc + 1e-10 * (pc == 0))
        pglc[torch.sum(pglc, dim=1, keepdim=False) == 0] = 1 / len(pglc[0])
        hglc_c = -torch.sum(torch.log(pglc + 1e-10 * (pglc == 0)) * pglc, dim=1)
        nmi_scores.append(torch.min(hglc_c) / hg)
    
    return nmi_scores


def compute_nmi(y_true, y_pred):
    """
    计算归一化互信息 (NMI)
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: NMI值
    """
    return normalized_mutual_info_score(y_true, y_pred)


def multi_mnce(pred_vec, group_vec_list, class_num, group_num_list):
    """
    计算多个敏感属性的 MNCE 指标
    """
    mnce_scores = []
    
    # 确保 pred_vec 是整数类型
    pred_vec = pred_vec.long()
    
    for group_vec, group_num in zip(group_vec_list, group_num_list):
        # 确保 group_vec 是整数类型
        group_vec = group_vec.long()
        
        # 计算混淆矩阵
        O = torch.zeros((class_num, group_num), device='cuda')
        for ci, gi in zip(pred_vec, group_vec):
            O[ci, gi] += 1
        
        # 添加调试信息
        print(f"混淆矩阵形状: {O.shape}")
        print(f"混淆矩阵非零元素数量: {torch.sum(O > 0).item()}")
        
        # 计算组别熵 H(G)
        group_probs = torch.sum(O, dim=0) / torch.sum(O)
        hg = -torch.sum(group_probs * torch.log(group_probs + 1e-10))
        
        # 计算条件熵 H(G|C)
        cluster_probs = torch.sum(O, dim=1) / torch.sum(O)
        hglc_c = torch.zeros(class_num, device='cuda')
        
        for c in range(class_num):
            if cluster_probs[c] > 0:
                cond_probs = O[c] / (torch.sum(O[c]) + 1e-10)
                hglc_c[c] = -torch.sum(cond_probs * torch.log(cond_probs + 1e-10))
        
        # 计算 MNCE = min(H(G|C)) / H(G)
        if hg > 0:
            mnce = torch.min(hglc_c) / hg
            mnce_scores.append(mnce.item())
        else:
            print("警告：组别熵为0，跳过此敏感属性的MNCE计算")
            mnce_scores.append(0.0)
    
    # 计算平均 MNCE
    avg_mnce = np.mean(mnce_scores) if mnce_scores else 0.0
    
    return mnce_scores, avg_mnce


def evaluate2(feature_vec, pred_vec, type_vec, group_vec_list, epoch=None, args=None):
    """
    评估聚类结果
    :param feature_vec: 特征向量
    :param pred_vec: 预测的聚类标签
    :param type_vec: 真实标签
    :param group_vec_list: 敏感属性列表
    :param epoch: 当前训练轮次
    :param args: 参数对象
    :return: 调整后的预测标签
    """
    global best_balance, best_nmi, best_ari, best_acc, BestCGF
    
    # 确保预测标签和真实标签从0开始且连续
    pred_vec_normalized = pred_vec - pred_vec.min()
    type_vec_normalized = type_vec - type_vec.min()
    
    # 确保所有组标签都是整数类型
    group_vec_list = [np.array(g, dtype=np.int64) for g in group_vec_list]
    
    # 检查预测标签和组标签是否为整数类型
    if not np.issubdtype(pred_vec_normalized.dtype, np.integer):
        pred_vec_normalized = pred_vec_normalized.astype(np.int64)
        print("已将预测标签转换为整数类型")
    if not all(np.issubdtype(g.dtype, np.integer) for g in group_vec_list):
        print("警告：某些组标签不是整数类型，这可能会影响评估结果")
    
    # 检查是否所有样本都被分配到同一个聚类
    if len(np.unique(pred_vec_normalized)) == 1:
        print("警告：所有样本都被分配到同一个聚类，这可能会导致评估结果不准确")
        # 如果使用了K-means，可以尝试重新聚类
        if 'kmeans' in globals():
            print("尝试使用K-means重新聚类...")
            try:
                kmeans = faiss.Kmeans(feature_vec.shape[1], len(np.unique(type_vec_normalized)))
                kmeans.train(feature_vec.astype(np.float32))
                _, pred_vec_normalized = kmeans.index.search(feature_vec.astype(np.float32), 1)
                pred_vec_normalized = pred_vec_normalized.flatten()
            except Exception as e:
                print(f"K-means重新聚类失败：{str(e)}")
    
    # 计算聚类平衡分数
    try:
        # 自动生成敏感属性名称
        sensitive_attr_names = [f"敏感属性{i+1}" for i in range(len(group_vec_list))]
        overall_balance_scores = []
        balance_scores = []
        for i, (group_vec, attr_name) in enumerate(zip(group_vec_list, sensitive_attr_names)):
            # 计算每个聚类的样本数量
            cluster_counts = np.bincount(pred_vec_normalized, minlength=len(np.unique(pred_vec_normalized)))
            # 计算每个敏感属性组在每个聚类中的样本数量
            group_counts = np.zeros((len(np.unique(pred_vec_normalized)), len(np.unique(group_vec))))
            for cluster in range(len(np.unique(pred_vec_normalized))):
                for group in range(len(np.unique(group_vec))):
                    group_counts[cluster, group] = np.sum((pred_vec_normalized == cluster) & (group_vec == group))
            # 计算每个聚类的平衡分数
            cluster_balance_scores = []
            for cluster in range(len(np.unique(pred_vec_normalized))):
                if cluster_counts[cluster] > 0:
                    group_ratios = group_counts[cluster] / cluster_counts[cluster]
                    balance_score = 1 - np.sum(np.abs(group_ratios - 1/len(np.unique(group_vec)))) / 2
                    cluster_balance_scores.append(balance_score * 100)
                else:
                    cluster_balance_scores.append(0)
            overall_balance = np.mean(cluster_balance_scores)
            balance_scores.append((cluster_balance_scores, overall_balance))
            overall_balance_scores.append(overall_balance)
        # 计算综合平衡分数（所有敏感属性的平均值）
        combined_balance = np.mean(overall_balance_scores)
    except Exception as e:
        print(f"计算平衡分数时出错：{str(e)}")
        # 使用简化的方法计算平衡分数
        sensitive_attr_names = [f"敏感属性{i+1}" for i in range(len(group_vec_list))]
        balance_scores = []
        overall_balance_scores = []
        for i, (group_vec, attr_name) in enumerate(zip(group_vec_list, sensitive_attr_names)):
            cluster_balance_scores = []
            for cluster in np.unique(pred_vec_normalized):
                cluster_mask = pred_vec_normalized == cluster
                if np.sum(cluster_mask) > 0:
                    group_counts = np.bincount(group_vec[cluster_mask], minlength=len(np.unique(group_vec)))
                    group_ratios = group_counts / np.sum(cluster_mask)
                    balance_score = 1 - np.sum(np.abs(group_ratios - 1/len(np.unique(group_vec)))) / 2
                    cluster_balance_scores.append(balance_score * 100)
                else:
                    cluster_balance_scores.append(0)
            overall_balance = np.mean(cluster_balance_scores)
            balance_scores.append((cluster_balance_scores, overall_balance))
            overall_balance_scores.append(overall_balance)
        combined_balance = np.mean(overall_balance_scores)
    # MNCE 评估
    avg_mnce = 0.0  # 初始化 MNCE 值
    try:
        pred_vec_tensor = torch.tensor(pred_vec_normalized, dtype=torch.long, device='cuda')
        group_vec_tensors = [torch.tensor(g, dtype=torch.long, device='cuda') for g in group_vec_list]
        mnce_scores, avg_mnce = multi_mnce(
            pred_vec=pred_vec_tensor,
            group_vec_list=group_vec_tensors,
            class_num=len(np.unique(pred_vec_normalized)),
            group_num_list=[len(np.unique(g)) for g in group_vec_list]
        )
        print("\n=== MNCE 评估结果 ===")
        for i, (score, attr_name) in enumerate(zip(mnce_scores, sensitive_attr_names)):
            print(f"敏感属性 {i+1} ({attr_name}) 的 MNCE: {score:.4f}")
        print(f"平均 MNCE: {avg_mnce*100:.2f}%")
        if epoch is not None and args is not None and epoch > args.WarmAll:
            global BestMnce
            if avg_mnce > BestMnce:
                BestMnce = avg_mnce
            print(f"最佳 MNCE: {BestMnce:.4f}")
    except Exception as e:
        print(f"计算 MNCE 时出错：{str(e)}")
        print(f"错误详情：{type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        avg_mnce = 0.0
    # 尝试调整预测标签以匹配真实标签
    try:
        # 使用匈牙利算法找到最佳匹配
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(np.unique(pred_vec_normalized)), len(np.unique(type_vec_normalized))))
        for i in range(len(np.unique(pred_vec_normalized))):
            for j in range(len(np.unique(type_vec_normalized))):
                cost_matrix[i, j] = -np.sum((pred_vec_normalized == i) & (type_vec_normalized == j))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 创建映射字典
        mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        
        # 应用映射
        pred_adjusted = np.array([mapping[p] for p in pred_vec_normalized])
    except Exception as e:
        print(f"调整预测标签时出错：{str(e)}")
        pred_adjusted = pred_vec_normalized
    
    # 计算评估指标
    try:
        nmi = normalized_mutual_info_score(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算NMI时出错：{str(e)}")
        nmi = 0
    
    try:
        acc = cluster_acc(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算准确率时出错：{str(e)}")
        acc = 0
    
    try:
        ari = adjusted_rand_score(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算ARI时出错：{str(e)}")
        ari = 0

    # 计算Fβ-Multi指标（NMI和Cross-MNCE的调和平均），默认β=1
    beta = 1.0
    try:
        # nmi和avg_mnce都在[0,1]区间
        if (nmi + avg_mnce) > 0:
            F_multi = (1 + beta ** 2) * nmi * avg_mnce / (beta ** 2 * nmi + avg_mnce)
        else:
            F_multi = 0.0
    except Exception as e:
        print(f"计算Fβ-Multi时出错：{str(e)}")
        F_multi = 0.0

    # 在计算完所有指标后，只有在预热阶段结束后才更新最佳结果
    if epoch is not None and args is not None and epoch > args.WarmAll:
        if combined_balance > best_balance:
            best_balance = combined_balance
        
        if nmi * 100 > best_nmi:
            best_nmi = nmi * 100
        
        if ari * 100 > best_ari:
            best_ari = ari * 100
        
        if acc * 100 > best_acc:
            best_acc = acc * 100
    
    # 打印当前结果
    print("\n=== 聚类评估结果 ===")
    print(f"综合平衡分数 (平均偏差法): {combined_balance:.2f}%")
    print(f"NMI: {nmi*100:.2f}%")
    print(f"ARI: {ari*100:.2f}%")
    print(f"准确率: {acc*100:.2f}%")
    print(f"Cross-MNCE: {avg_mnce*100:.2f}%")
    print(f"Fβ-Multi(NMI, Cross-MNCE): {F_multi*100:.2f}%")
    
    # === KNN无偏互信息估计（NPEET） ===
    try:
        print("\n=== KNN无偏互信息估计（NPEET） ===")
        # 单独属性
        mi_list = []
        for g in group_vec_list:
            mi = ee.mi(g.reshape(-1, 1), pred_adjusted.reshape(-1, 1))
            if mi < 0:
                mi = 0.0
            mi_list.append(mi)
        mi_sum = sum(mi_list)
        for i, mi in enumerate(mi_list):
            print(f"敏感属性{i+1}与聚类的互信息(KNN法): {mi:.4f}")
        print(f"互信息之和: {mi_sum:.4f}")
        # 组合属性
        G_combo = np.stack(group_vec_list, axis=1)
        mi_combo = ee.mi(G_combo, pred_adjusted.reshape(-1, 1))
        if mi_combo < 0:
            mi_combo = 0.0
        print(f"组合属性与聚类的互信息(KNN法): {mi_combo:.4f}")
        delta = mi_sum - mi_combo
        if delta < 0:
            delta = 0.0
        print(f"交叉影响量（delta）: {delta:.4f}")
        if delta > 0:
            print("实证：存在交叉影响！（KNN法）")
        else:
            print("未观测到交叉影响。（KNN法）")
        # 记录和输出历史最优delta
        global BestDeltaKNN
        if delta > BestDeltaKNN:
            BestDeltaKNN = delta
        print(f"历史最优交叉影响量（delta，KNN法）: {BestDeltaKNN:.4f}")
    except Exception as e:
        print(f"KNN无偏互信息估计计算出错: {e}")
    
    # 只有在预热阶段结束后才打印最佳结果
    if epoch is not None and args is not None and epoch > args.WarmAll:
        print("\n=== 最佳评估结果 ===")
        print(f"最佳平衡分数 (平均偏差法): {best_balance:.2f}%")
        print(f"最佳NMI: {best_nmi:.2f}%")
        print(f"最佳ARI: {best_ari:.2f}%")
        print(f"最佳准确率: {best_acc:.2f}%")
        print(f"最佳Cross-MNCE: {BestMnce*100:.2f}%")
        # 用历史最优NMI和Cross-MNCE计算Fβ-Multi
        best_nmi_frac = best_nmi / 100
        best_mnce_frac = BestMnce
        if (best_nmi_frac + best_mnce_frac) > 0:
            F_multi_best = (1 + beta ** 2) * best_nmi_frac * best_mnce_frac / (beta ** 2 * best_nmi_frac + best_mnce_frac)
        else:
            F_multi_best = 0.0
        print(f"最佳Fβ-Multi(NMI, Cross-MNCE): {F_multi_best*100:.2f}%")
        
        print(f"最佳CGF: {BestCGF:.6f}")
    
    # 打印聚类分布
    print("\n=== 聚类分布（使用所有数据）===")
    unique_labels, counts = np.unique(pred_vec_normalized, return_counts=True)
    total_samples = len(pred_vec_normalized)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"聚类 {label}: {count} 个样本 ({percentage:.2f}%)")
    
    print("真实标签类别:", np.unique(type_vec_normalized), "长度:", len(type_vec_normalized))
    print("聚类标签类别:", np.unique(pred_adjusted), "长度:", len(pred_adjusted))
    
    # === CGF（交叉群体公平性）指标 ===
    try:
        print("\n=== CGF（交叉群体公平性）指标 ===")
        # 1. 构造交叉组别标签
        cross_group = list(zip(*group_vec_list))
        cross_group = np.array([str(t) for t in cross_group])
        unique_cross_groups = np.unique(cross_group)
        unique_clusters = np.unique(pred_adjusted)
        N = len(pred_adjusted)
        cgf = 0.0
        for t in unique_cross_groups:
            idx_t = (cross_group == t)
            size_t = np.sum(idx_t)
            p_t = size_t / N
            for k in unique_clusters:
                idx_k = (pred_adjusted == k)
                size_k = np.sum(idx_k)
                if size_k == 0:
                    continue
                p_tk = np.sum(idx_t & idx_k) / size_k
                cgf += (p_tk - p_t) ** 2
        print(f"CGF: {cgf:.6f}")
        # 记录和输出历史最优CGF
        if BestCGF is None or cgf < BestCGF:
            BestCGF = cgf
    except Exception as e:
        print(f"CGF计算出错: {e}")
    
    # 每50个epoch记录一次最佳结果到文件
    if epoch is not None and args is not None and (epoch + 1) % 50 == 0:
        results_dir = 'Results'
        os.makedirs(results_dir, exist_ok=True)
        results_file_path = os.path.join(results_dir, f'{args.dataset}_results.txt')

        # 重新计算最佳Fβ-Multi
        best_nmi_frac = best_nmi / 100
        best_mnce_frac = BestMnce
        if (best_nmi_frac + best_mnce_frac) > 0:
            F_multi_best = (1 + beta ** 2) * best_nmi_frac * best_mnce_frac / (beta ** 2 * best_nmi_frac + best_mnce_frac)
        else:
            F_multi_best = 0.0

        with open(results_file_path, 'a') as f:
            f.write(f"--- Epoch {epoch + 1} ---\n")
            f.write(f"最佳平衡分数 (平均偏差法): {best_balance:.2f}%\n")
            f.write(f"最佳NMI: {best_nmi:.2f}%\n")
            f.write(f"最佳ARI: {best_ari:.2f}%\n")
            f.write(f"最佳准确率: {best_acc:.2f}%\n")
            f.write(f"最佳Cross-MNCE: {BestMnce*100:.2f}%\n")
            f.write(f"最佳Fβ-Multi(NMI, Cross-MNCE): {F_multi_best*100:.2f}%\n")
            if BestCGF is not None:
                f.write(f"最佳CGF (越小越好): {BestCGF:.6f}\n")
            else:
                f.write("最佳CGF (越小越好): N/A\n")
            f.write(f"最佳交叉影响量（delta，KNN法）: {BestDeltaKNN:.4f}\n")
            f.write("\n")
        
        print(f"\n已将第 {epoch + 1} 轮的最佳结果记录到 {results_file_path}")
    
    return pred_adjusted


def entroph(v):
    """
    计算熵
    :param v: 概率分布向量
    :return: 熵值
    """
    # 避免log(0)
    v_safe = v + 1e-10 * (v == 0)
    return -torch.sum(torch.log(v_safe) * v)


def cluster_metrics(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(label, pred_adjusted)
    # acc = 0
    return nmi, ari, acc, pred_adjusted


def fair_metrics(feature_vec, batch_vec, pred_vec, type_vec):
    pass


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true,
                                                cluster_assignments,
                                                labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    pred_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = pred_to_true_cluster_labels[cluster_assignments]
    return np.asarray(y_pred, dtype=int)


def UMAP(feature_vec, type_vec, group_vec_list, pred_vec, n_type, n_batch, args, epoch):
    t = time.time()
    sc.set_figure_params(figsize=(4, 4), dpi=300)

    adata = sc.AnnData(feature_vec)
    sc.pp.neighbors(adata)
    adata.obs['cluster'] = pd.DataFrame(pred_vec).values.astype(np.str_)
    adata.obs['type'] = pd.DataFrame(type_vec).values.astype(np.str_)
    
    # 添加两个敏感属性
    adata.obs['group_source'] = pd.DataFrame(group_vec_list[0]).values.astype(np.str_)  # 数据源
    adata.obs['group_brightness'] = pd.DataFrame(group_vec_list[1]).values.astype(np.str_)  # 亮度

    sc.tl.umap(adata)
    
    # 绘制聚类结果
    sc.pl.umap(adata,
               color=['cluster'],
               palette=sns.color_palette("husl", n_type),
               save='E{:03d}UmapCluster{}.png'.format(epoch, str(args.dataset)),
               show=False)
    
    # 绘制真实标签
    sc.pl.umap(adata,
               color=['type'],
               palette=sns.color_palette("husl", n_type),
               save='E{:03d}UmapType{}.png'.format(epoch, str(args.dataset)),
               show=False)
    
    # 绘制数据源分布
    sc.pl.umap(adata,
               color=['group_source'],
               palette=sns.color_palette("hls", n_batch),
               save='E{:03d}UmapSource{}.png'.format(epoch, str(args.dataset)),
               show=False)
    
    # 绘制亮度分布
    sc.pl.umap(adata,
               color=['group_brightness'],
               palette=sns.color_palette("hls", 2),  # 亮度只有两组
               save='E{:03d}UmapBrightness{}.png'.format(epoch, str(args.dataset)),
               show=False)
    
    # 移动图片到Visualization目录
    roott = './figures/'
    for root, dirs, files in os.walk(roott):
        for f in files:
            FileOperator(
                os.path.join(root, f)
            ).rename(
                os.path.join('../Visualization', f.replace('umapE', 'E')),
                auto_rename=True
            )
    
    if PrintTimer:
        print('VisualizeScatter finished with in {:.03f} seconds (x.shape == {}).'.format(
            time.time() - t,
            feature_vec.shape,
        ))


def main():
    # pred_vec = (np.arange(300) + 10) % 30
    # pred_vec = (np.arange(10)+10)%3
    # type_vec = np.arange(300) % 30
    # group_vec = np.arange(300) % 2
    # relative_fairness()

    # data = np.load('/mnt/18t/pengxin/Codes/0412/RunSet-0419_Load/Office_NetShuffleLikGDecFeaTanActSigWarmAll20InfoBalanceLoss0.05InfoFairLoss0.20Tb0.10OneHot0.05Th0.10_G0B0512torch1110/NpPoints/Np001.npz')
    data = np.load('/mnt/18t/pengxin/Checkpoints/FairClustering/FCMI/SotaNp/MouseAtlas/Np079.npz')
    feature_vec = data['feature_vec']
    type_vec = data['type_vec']
    group_vec = data['group_vec']
    pred_vec = data['pred_vec']
    epoch = data['epoch']
    
    # 创建一个简单的Args对象
    class Args:
        def __init__(self):
            self.WarmAll = 20  # 设置一个默认值
    
    args = Args()
    evaluate2(None, pred_vec, type_vec, [group_vec], epoch=epoch, args=args)
    # print(nmi)
    # roott = './figures/umap/'
    # for root, dirs, files in os.walk(roott):
    #     for f in files:
    #         print(os.path.join('../Visualization', f))
    #         FileOperator(
    #             os.path.join(root, f)
    #         ).rename(
    #             os.path.join('../Visualization', f)
    #         )


if __name__ == '__main__':
    main()

def cluster_acc(y_true, y_pred):
    """
    计算聚类准确率
    :param y_true: 真实标签
    :param y_pred: 预测标签（已通过匈牙利算法最佳匹配）
    :return: 准确率
    """
    return accuracy_score(y_true, y_pred)

def evaluate_fcn(feature_vec, pred_vec, type_vec, group_vec_list):
    """
    专门用于评估NetFCN网络和MTFL数据集的评估函数
    :param feature_vec: 特征向量
    :param pred_vec: 预测的聚类标签
    :param type_vec: 真实标签
    :param group_vec_list: 敏感属性列表，每个元素是一个一维数组
    :return: 调整后的预测标签
    """
    # 确保预测标签从0开始且连续
    pred_vec_normalized = pred_vec - np.min(pred_vec)
    type_vec_normalized = type_vec - np.min(type_vec)
    
    # 计算平衡分数
    balance_scores = []
    sensitive_attr_names = ["眼镜属性", "微笑属性"]
    overall_balance_scores = []
    
    for i, (group_vec, attr_name) in enumerate(zip(group_vec_list, sensitive_attr_names)):
        cluster_balance_scores = []
        for cluster in np.unique(pred_vec_normalized):
            cluster_mask = pred_vec_normalized == cluster
            if np.sum(cluster_mask) > 0:
                # 确保group_vec是一维数组
                group_vec_flat = group_vec.flatten()
                group_counts = np.bincount(group_vec_flat[cluster_mask], minlength=len(np.unique(group_vec_flat)))
                group_ratios = group_counts / np.sum(cluster_mask)
                balance_score = 1 - np.sum(np.abs(group_ratios - 1/len(np.unique(group_vec_flat)))) / 2
                cluster_balance_scores.append(balance_score * 100)
            else:
                cluster_balance_scores.append(0)
        overall_balance = np.mean(cluster_balance_scores)
        balance_scores.append((cluster_balance_scores, overall_balance))
        overall_balance_scores.append(overall_balance)
    
    combined_balance = np.mean(overall_balance_scores)
    
    # 尝试调整预测标签以匹配真实标签
    try:
        # 使用匈牙利算法找到最佳匹配
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(np.unique(pred_vec_normalized)), len(np.unique(type_vec_normalized))))
        for i in range(len(np.unique(pred_vec_normalized))):
            for j in range(len(np.unique(type_vec_normalized))):
                cost_matrix[i, j] = -np.sum((pred_vec_normalized == i) & (type_vec_normalized == j))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 创建映射字典
        mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        
        # 应用映射
        pred_adjusted = np.array([mapping[p] for p in pred_vec_normalized])
    except Exception as e:
        print(f"调整预测标签时出错：{str(e)}")
        pred_adjusted = pred_vec_normalized
    
    # 计算评估指标
    try:
        nmi = normalized_mutual_info_score(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算NMI时出错：{str(e)}")
        nmi = 0
    
    try:
        acc = cluster_acc(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算准确率时出错：{str(e)}")
        acc = 0
    
    try:
        ari = adjusted_rand_score(type_vec_normalized, pred_adjusted)
    except Exception as e:
        print(f"计算ARI时出错：{str(e)}")
        ari = 0
    
    # 打印结果
    print("\n=== MTFL数据集聚类评估结果 ===")
    print(f"综合平衡分数: {combined_balance:.2f}%")
    print(f"NMI: {nmi*100:.2f}%")
    print(f"ARI: {ari*100:.2f}%")
    print(f"准确率: {acc*100:.2f}%")
    
    # # 打印每个敏感属性的平衡分数
    # print("\n=== 敏感属性平衡详情 ===")
    # for i, ((cluster_scores, overall_score), attr_name) in enumerate(zip(balance_scores, sensitive_attr_names)):
    #     print(f"\n敏感属性 {i+1} ({attr_name}) 的聚类平衡分数:")
    #     for j, score in enumerate(cluster_scores):
    #         print(f"  聚类 {j}: {score:.2f}%")
    #     print(f"  整体平衡分数: {overall_score:.2f}%")
    
    # 打印聚类分布
    print("\n=== 聚类分布 ===")
    unique_labels, counts = np.unique(pred_vec_normalized, return_counts=True)
    total_samples = len(pred_vec_normalized)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"聚类 {label}: {count} 个样本 ({percentage:.2f}%)")
    
    return pred_adjusted
