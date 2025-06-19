import itertools
# import time
# import warnings

# import numpy
import math
import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Experiment.Visualize import tsne, visualize_scatter, visual_matrix_console
from Utils import TimeOperator, DirectoryOperator
from DataSetMaster.dataset import get_clusters
from evaluate import UMAP, evaluate2
import evaluate
from loss import instance_contrastive_loss
import loss as loss_model
import faiss
from MainLauncher import LimitKmeans, CodeTest, DrawMax
from tqdm import tqdm
import pandas as pd


##其主要功能是对输入的特征向量进行降维处理，并将降维后的特征向量以散点图的形式进行可视化
def visualize(feature_vec, type_vec, group_vec, pred_vec, epoch):
    vis_fea = tsne(feature_vec)
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Type.jpg'.format(epoch),
                      label_color=type_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Cluster.jpg'.format(epoch),
                      label_color=pred_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Group.jpg'.format(epoch),
                      label_color=group_vec,
                      label_shape=type_vec,
                      )


def show_distribution_ct(type_vec, group_vec, pred_vec, class_num, group_num):
    v = np.zeros((class_num, class_num, group_num), dtype=int)
    for t, c, g in zip(type_vec, pred_vec, group_vec):
        v[t, c, g] += 1
    visual_matrix_console(x=v)


# if __name__ == '__main__':
#     gn = 3
#     t = np.arange(1000)
#     typ = t % 10
#     pred = t * t % 10
#     gr = t % gn
#     show_distribution_ct(type_vec=typ, group_vec=gr, pred_vec=pred, class_num=10, group_num=gn)


def inference(net, test_dataloader, args):
    net.eval()
    feature_vec, type_vec, group_vec_list, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, g, y, idx) in test_dataloader:
            x = x.cuda()
            g = [g_i.cuda() for g_i in g]  # g是一个列表，包含两个敏感属性张量
            h = net.encode(x, g)
            c = net.encode_class(h).detach()
            pred = torch.argmax(c, dim=1)
            feature_vec.extend(h[0].detach().cpu().numpy())
            type_vec.extend(y.cpu().numpy())
            
            # 收集敏感属性
            if len(group_vec_list) == 0:
                group_vec_list = [[] for _ in range(len(g))]
            for i, g_i in enumerate(g):
                # 确保g_i是一维数组
                if len(g_i.shape) > 1:
                    g_i = g_i.squeeze()
                # 如果g_i是标量，将其转换为数组
                if g_i.dim() == 0:
                    g_i = g_i.unsqueeze(0)
                group_vec_list[i].extend(g_i.cpu().numpy())
            pred_vec.extend(pred.cpu().numpy())
    
    feature_vec = np.array(feature_vec)
    type_vec = np.array(type_vec)
    # 将group_vec_list转换为numpy数组，确保每个敏感属性都是一维数组
    group_vec_list = [np.array(g).flatten() for g in group_vec_list]
    pred_vec = np.array(pred_vec)
    
    d = net.representation_dim
    
    # 新增：根据args.cluster_method选择聚类算法
    if hasattr(args, 'cluster_method'):
        cluster_method = args.cluster_method
    else:
        cluster_method = 'kmeans'
    
    if cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=net.class_num, n_init=10, max_iter=300, random_state=0)
        pred_labels = kmeans.fit_predict(feature_vec)
        centroids = kmeans.cluster_centers_
    elif cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=net.class_num, covariance_type='full', random_state=0)
        pred_labels = gmm.fit_predict(feature_vec)
        centroids = gmm.means_
    elif cluster_method == 'spectral':
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=net.class_num, affinity='nearest_neighbors', random_state=0)
        pred_labels = spectral.fit_predict(feature_vec)
        centroids = None  # 谱聚类没有中心
    elif cluster_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        pred_labels = dbscan.fit_predict(feature_vec)
        centroids = None  # DBSCAN没有中心
    else:
        raise NotImplementedError(f'未知聚类方法: {cluster_method}')
    
    # 用pred_labels替换原pred_vec
    pred_vec = pred_labels
    
    # 兼容后续centroids为None的情况
    net.train()
    if centroids is not None:
        print(f"Centroids shape: {centroids.shape}")
    else:
        print("本次聚类方法无聚类中心（centroids=None）")
    
    # 打印敏感属性的形状
    print("Sensitive attributes shapes:")
    for i, g in enumerate(group_vec_list):
        print(f"  Attribute {i}: {g.shape}")
    
    return feature_vec, type_vec, group_vec_list, pred_vec, centroids


def show_distribution(cluster_vec, group_vec, class_num, group_num):
    for it in np.arange(group_num):
        print('{:4d}, '.format(it), end='')
    print('')
    cluster_group = torch.zeros((class_num, group_num), dtype=torch.int)
    for i, j in zip(cluster_vec, group_vec):
        cluster_group[i, j] += 1
    # cluster_group = cluster_group[torch.argsort(torch.sum(cluster_group, dim=1))]
    for line in cluster_group:
        print('{:4d}: '.format(torch.sum(line)), end='')
        for it in line:
            print('{:4d}, '.format(it), end='')
        print('')


def save_checkpoint(state, epoch, dataset_name):
    """
    it has been trained for *epoch* epochs
    """
    filename = 'Epoch{:03d}_{}.checkpoint'.format(epoch, dataset_name)
    checkpoint_dir = os.path.join(
        os.path.dirname(os.getcwd()),
        'Checkpoints',
        filename
    )
    DirectoryOperator.FoldOperator(directory=checkpoint_dir).make_fold()
    if os.path.exists(checkpoint_dir):
        warnings.warn('Checkpoint exist and been replaced.({})'.format(checkpoint_dir))
    print('Save check point into {}'.format(checkpoint_dir))
    torch.save(state, checkpoint_dir)


class Net(nn.Module):
    def __init__(self, class_num, group_nums, args, num_sensitive_attrs=2):
        super(Net, self).__init__()
        self.class_num = class_num
        if args.representation_dim > 0:
            self.representation_dim = args.representation_dim
        else:
            self.representation_dim = class_num
        self.group_nums = group_nums  # 传入每个敏感属性的类别数列表
        self.encoder_out_dim = 784
        self.args = args
        self.num_sensitive_attrs = num_sensitive_attrs
        
        # 初始化聚类中心
        self.cluster_centers = torch.randn(self.class_num, self.representation_dim).cuda()
        if self.args.RepresentationType == 'Normalize':
            self.cluster_centers = F.normalize(self.cluster_centers, dim=1)
        
        # 为每个敏感属性创建独立的编码器
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                for _ in range(self.num_sensitive_attrs if args.GroupWiseEncoder else 1)
            ]
        )
        
        self.encoder_linear = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
                for _ in range(self.num_sensitive_attrs if args.WithFeatureB else 1)
            ]
        )
        
        self.decoder_linear = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.representation_dim, 256), 
                    nn.ReLU(), 
                    nn.Linear(256, self.encoder_out_dim), 
                    nn.ReLU(),
                )
                for _ in range(self.num_sensitive_attrs if args.WithFeatureB else 1)
            ]
        )
        
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        32 if args.WithFeatureB else 16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        32, 32, kernel_size=3, stride=1, padding=1, output_padding=0
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                )
                for _ in range(self.num_sensitive_attrs if args.GroupWiseDecoder else 1)
            ]
        )
        
        # 为每个敏感属性创建独立的判别器
        self.discriminator_foreground = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(), 
                nn.Linear(self.representation_dim, 512),
                nn.BatchNorm1d(512), 
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256), 
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2), 
                nn.Linear(256, 1)
            )
            for _ in range(self.num_sensitive_attrs)
        ])
        self.initialize_weights(self.discriminator_foreground)
        
        # 注意力机制（仅多于3个敏感属性时启用）
        if self.num_sensitive_attrs > 3:
            self.attention_weights = nn.Parameter(torch.ones(self.num_sensitive_attrs) / self.num_sensitive_attrs)
            self.attention_weights.requires_grad = True

    def initialize_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def initialize_cluster_centers(self, feature_vec):
        """
        使用K-means初始化聚类中心
        """
        if self.cluster_centers is None:
            # 使用K-means初始化
            kmeans = faiss.Kmeans(
                self.representation_dim,
                self.class_num,
                niter=100,
                verbose=False
            )
            kmeans.train(feature_vec.astype(np.float32))
            centers = kmeans.centroids
            
            # 转换为tensor并归一化
            self.cluster_centers = torch.from_numpy(centers).float().cuda()
        if self.args.RepresentationType == 'Normalize':
                self.cluster_centers = F.normalize(self.cluster_centers, dim=1)
            
        print(f"Initialized cluster centers with shape: {self.cluster_centers.shape}")
        print(f"Number of classes: {self.class_num}")
        print(f"Representation dimension: {self.representation_dim}")

    def encode(self, x, group_indices):
        if self.args.GroupWiseEncoder:
            hh = torch.zeros((len(x), self.encoder_out_dim), device='cuda')
            g = group_indices[0]
            for g_val in torch.unique(g):
                # 将张量转换为整数
                g_val_int = int(g_val.item())
                ind = g_val == g
                hh[ind] = self.encoder[g_val_int](x[ind])
        else:
            hh = self.encoder[0](x)
        
        # print(f"Encoder output shape: {hh.shape}")
        
        a = self.encoder_linear[0](hh)
        # print(f"Encoder linear output shape: {a.shape}")
        
        if self.args.RepresentationType == 'Normalize':
            a = F.normalize(a, dim=1)
        elif self.args.RepresentationType == 'None':
            pass
        elif self.args.RepresentationType == 'Relu':
            pass
        else:
            raise NotImplementedError('RepresentationType')
        
        if self.args.WithFeatureB:
            b = self.encoder_linear[1](hh)
            if self.args.RepresentationType == 'Normalize':
                b = F.normalize(b, dim=1)
            elif self.args.RepresentationType == 'None':
                pass
            elif self.args.RepresentationType == 'Relu':
                b = nn.ReLU()(b)
            else:
                raise NotImplementedError('RepresentationType')
        else:
            b = None
            
        return a, b

    def encode_class(self, h):
        if isinstance(h, tuple):
            h = h[0]
        if len(h.shape) == 1:
            h = h.unsqueeze(0)
        if self.args.RepresentationType == 'Normalize':
            c = h @ self.cluster_centers.T
        elif self.args.RepresentationType == 'None':
            c = -torch.cdist(h, self.cluster_centers) ** 2 / 700
        elif self.args.RepresentationType == 'Relu':
            c = -torch.cdist(h, self.cluster_centers) ** 2 / 700
        else:
            raise NotImplementedError('RepresentationType')
        return c

    def decode(self, z, group_indices):
        if self.args.WithFeatureB:
            z = torch.cat([dec(zi) for zi, dec in zip(z, self.decoder_linear)], dim=1)
        else:
            z = self.decoder_linear[0](z[0])
        
        # 重塑张量以适应卷积操作
        batch_size = z.size(0)
        z = z.view(batch_size, 16, 7, 7)  # 784 = 16 * 7 * 7
        
        if self.args.GroupWiseDecoder:
            x_ = torch.zeros((len(z), 1, 28, 28), device='cuda')
            g = group_indices[1]  # 使用第二个敏感属性
            for g_val in torch.unique(g):
                ind = g_val == g
                if torch.sum(ind) == 0:
                    pass
                elif torch.sum(ind) == 1:
                    x_[ind] = self.decoder[g_val](torch.cat([z[ind], z[ind]], dim=1))[[0]]
                else:
                    x_[ind] = self.decoder[g_val](z[ind])
        else:
            x_ = self.decoder[0](z)
        return x_

    def get_attention_weights(self):
        if self.num_sensitive_attrs > 3:
            return F.softmax(self.attention_weights, dim=0)
        else:
            return None

    def group_wise_balance_loss(self, z, g):
        if self.num_sensitive_attrs <= 3:
            return self._group_wise_balance_loss_single(z, g[0], self.group_nums[0])
        else:
            losses = []
            for i, g_i in enumerate(g):
                group_num = self.group_nums[i]
                losses.append(self._group_wise_balance_loss_single(z, g_i, group_num))
            weights = self.get_attention_weights() if hasattr(self, 'get_attention_weights') else None
            if weights is not None:
                return sum(w * l for w, l in zip(weights, losses))
            else:
                return sum(losses) / len(losses)

    def _group_wise_balance_loss_single(self, z, g, group_num):
        probs = F.softmax(self.encode_class(z), dim=1)
        O = torch.zeros((self.class_num, group_num), device=z.device)
        E = torch.zeros((self.class_num, group_num), device=z.device)
        for b in range(group_num):
            O[:, b] = torch.sum(probs[g == b], dim=0)
            E[:, b] = (g == b).sum()
        O = O / (torch.sum(O, dim=1, keepdim=True) + 1e-10)
        E = E / (torch.sum(E, dim=1, keepdim=True) + 1e-10)
        loss = F.kl_div(torch.log(O + 1e-10), E, reduction='batchmean')
        return loss

    def cluster_wise_balance_loss(self, z, g):
        if self.num_sensitive_attrs <= 3:
            return self._cluster_wise_balance_loss_single(z, g[0], self.group_nums[0])
        else:
            losses = []
            for i, g_i in enumerate(g):
                group_num = self.group_nums[i]
                losses.append(self._cluster_wise_balance_loss_single(z, g_i, group_num))
            weights = self.get_attention_weights() if hasattr(self, 'get_attention_weights') else None
            if weights is not None:
                return sum(w * l for w, l in zip(weights, losses))
            else:
                return sum(losses) / len(losses)

    def _cluster_wise_balance_loss_single(self, z, g, group_num):
        probs = F.softmax(self.encode_class(z), dim=1)
        O = torch.zeros((self.class_num, group_num), device=z.device)
        E = torch.zeros((self.class_num, group_num), device=z.device)
        for b in range(group_num):
            O[:, b] = torch.sum(probs[g == b], dim=0)
            E[:, b] = (g == b).sum()
        O = O / (torch.sum(O, dim=0, keepdim=True) + 1e-10)
        E = E / (torch.sum(E, dim=0, keepdim=True) + 1e-10)
        O = O.t()
        E = E.t()
        loss = F.kl_div(torch.log(O + 1e-10), E, reduction='batchmean')
        return loss

    def onehot_loss(self, z, g):
        """
        计算one-hot损失，使聚类结果更加明确
        :param z: 特征向量
        :param g: 敏感属性
        :return: 损失值
        """
        probs = F.softmax(self.encode_class(z), dim=1)
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))

    def consistency_loss(self, z, g):
        """
        计算一致性损失，确保相似样本的聚类结果相似
        :param z: 特征向量
        :param g: 敏感属性
        :return: 损失值
        """
        probs = F.softmax(self.encode_class(z), dim=1)
        # 计算样本间的相似度矩阵
        similarity = torch.mm(z, z.t())
        similarity = F.softmax(similarity, dim=1)
        
        # 计算聚类结果的一致性
        consistency = torch.mm(probs, probs.t())
        loss = F.mse_loss(consistency, similarity)
        return loss

    def proto_loss(self, z, g):
        """
        计算原型损失，使每个聚类的特征更加紧凑
        :param z: 特征向量
        :param g: 敏感属性
        :return: 损失值
        """
        probs = F.softmax(self.encode_class(z), dim=1)
        # 计算每个聚类的中心
        centers = torch.mm(probs.t(), z) / (torch.sum(probs, dim=0, keepdim=True).t() + 1e-10)
        
        # 计算每个样本到其聚类中心的距离
        dist = torch.cdist(z, centers)
        loss = torch.mean(torch.sum(probs * dist, dim=1))
        return loss

    def intersectional_balance_loss(self, predicted_probs, group_vec_list):
        """
        计算基于"最差比率法"的交叉平衡损失。
        损失 = 1 - 交叉平衡分数。
        """
        # 1. 创建交叉属性ID
        groups_np = [g.cpu().numpy() for g in group_vec_list]
        cross_group_tuples = list(zip(*groups_np))
        intersectional_g_numeric, _ = pd.factorize(cross_group_tuples)
        intersectional_g = torch.from_numpy(intersectional_g_numeric).to(predicted_probs.device)
        group_num = len(np.unique(intersectional_g_numeric))
        class_num = self.class_num

        # 2. 计算软计数的混淆矩阵
        # O 的形状为 (class_num, group_num)
        O = torch.zeros((class_num, group_num), device=predicted_probs.device)
        for g_val in range(group_num):
            mask = (intersectional_g == g_val)
            if mask.sum() > 0:
                O[:, g_val] = torch.sum(predicted_probs[mask], dim=0)

        # 避免除以零
        O[O == 0] = 1e-5

        # 3. 计算 min/max 比率
        balance_per_cluster = torch.amin(O, dim=1) / torch.amax(O, dim=1)
        
        # 4. 计算最终的平衡分数（木桶效应）
        final_balance_score = torch.amin(balance_per_cluster)

        # 返回损失
        return 1.0 - final_balance_score

    def update_cluster_center(self, centers):
        """
        更新聚类中心
        """
        new_centers = torch.from_numpy(centers).float().cuda()
        if self.args.RepresentationType == 'Normalize':
            self.cluster_centers = F.normalize(new_centers, dim=1)
        else:
            self.cluster_centers = new_centers
        
        print(f"Updated cluster centers shape: {self.cluster_centers.shape}")
        
        # 检查聚类中心是否有问题
        if torch.isnan(self.cluster_centers).any() or torch.isinf(self.cluster_centers).any():
            print("警告：聚类中心包含NaN或Inf值，使用随机初始化替代")
            self.cluster_centers = torch.randn_like(self.cluster_centers)
            if self.args.RepresentationType == 'Normalize':
                self.cluster_centers = F.normalize(self.cluster_centers, dim=1)

    def run(self, epochs, train_dataloader, test_dataloader, args):
        if args.loss_self_cons:
            clusters = get_clusters(args=args)

        optimizer_g = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.encoder_linear.parameters(),
                self.decoder_linear.parameters(),
                self.decoder.parameters(),
            ),
            lr=args.LearnRate,
            betas=(args.betas_a, args.betas_v),
            weight_decay=args.WeightDecay
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator_foreground.parameters(),
            lr=args.LearnRate,
            betas=(args.betas_a, args.betas_v),
            weight_decay=args.WeightDecay
        )
        mse_loss = nn.MSELoss().cuda()
        ce_loss = nn.CrossEntropyLoss().cuda()
        timer_all = TimeOperator.TimeOperator()
        timer_train = TimeOperator.TimeOperator()
        timer_infer = TimeOperator.TimeOperator()
        type_detail_shown = False
        start_epoch = 0
        
        # 加载检查点
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                self.load_state_dict(checkpoint['state_dict'])
                optimizer_g.load_state_dict(checkpoint['optimizer']['optimizer_g'])
                optimizer_d.load_state_dict(checkpoint['optimizer']['optimizer_d'])
                self.__dict__ = checkpoint['self_dic']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                epochs = start_epoch + 1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # 使用tqdm创建进度条
        epoch_pbar = tqdm(range(start_epoch, epochs), desc='Training Progress')
        for epoch in epoch_pbar:
            if (epoch + 1) == epochs or (epoch + 1) == 2 or (epoch + 1) == args.WarmAll:
                dct = {
                    'epoch'     : epoch,
                    'state_dict': self.state_dict(),
                    'optimizer' : {'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict()},
                }
                dct = {**dct, 'self_dic': self.__dict__}
                save_checkpoint(dct, epoch=epoch, dataset_name=args.dataset)
            
            # 学习率调整
            if (epoch + 1) <= args.LearnRateWarm:
                lr = args.LearnRate * (epoch + 1) / args.LearnRateWarm
            else:
                if args.LearnRateDecayType == 'None':
                    lr = args.LearnRate
                elif args.LearnRateDecayType == 'Exp':
                    lr = args.LearnRate * ((1 + 10 * (epoch + 1 - args.LearnRateWarm) / (
                            args.train_epoch - args.LearnRateWarm)) ** -0.75)
                elif args.LearnRateDecayType == 'Cosine':
                    lr = args.LearnRate * 0.5 * (1. + math.cos(
                        math.pi * (epoch + 1 - args.LearnRateWarm) / (args.train_epoch - args.LearnRateWarm)))
                else:
                    raise NotImplementedError('args.LearnRateDecayType')
                
            if lr != args.LearnRate:
                def adjust_learning_rate(optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_g)
                adjust_learning_rate(optimizer_d)

            timer_all.time()
            timer_infer.time()
            
            # 在第一次运行时初始化聚类中心
            if self.cluster_centers is None:
                print("Initializing cluster centers...")
                feature_vec, type_vec, group_vec_list, pred_vec, _ = inference(
                    self, test_dataloader, args
                )
                self.initialize_cluster_centers(feature_vec)
            
            # 推理阶段
            feature_vec, type_vec, group_vec_list, pred_vec, centers = inference(
                self, test_dataloader, args
            )
            

            if epoch % 1 == 0 and centers is not None:
                self.update_cluster_center(centers)
            else:
                if centers is None:
                    print("本次聚类方法无聚类中心，跳过update_cluster_center。")

            
            pred_adjusted = evaluate.evaluate2(feature_vec, pred_vec, type_vec, group_vec_list, epoch=epoch, args=args)
            
            # 在训练开始时和每50个轮次生成可视化图
            if epoch == 0 or (epoch + 1) % 50 == 0:
                print(f"\n生成第 {epoch + 1} 轮的可视化图...")
                # 确保Results目录存在
                os.makedirs('Results', exist_ok=True)
                # 生成可视化图
                evaluate.UMAP(
                    feature_vec=feature_vec,
                    type_vec=type_vec,
                    group_vec_list=group_vec_list,
                    pred_vec=pred_adjusted,
                    n_type=len(np.unique(type_vec)),
                    n_batch=len(np.unique(group_vec_list[0])),
                    args=args,
                    epoch=epoch + 1
                )
                # 将生成的图片移动到Results目录
                for root, dirs, files in os.walk('./figures'):
                    for f in files:
                        if f.endswith('.png'):
                            src_path = os.path.join(root, f)
                            dst_path = os.path.join('Results', f)
                            os.rename(src_path, dst_path)
                print(f"可视化图已保存到 Results 目录")
            
            if (epoch + 1) == epochs or (epoch + 1) == 2 or (epoch + 1) % max(1, args.WarmAll) == 0:
                np_dir = '../NpPoints/Np{:03d}'.format(epoch)
                DirectoryOperator.FoldOperator(directory=np_dir).make_fold()
                np.savez(np_dir, 
                        feature_vec=feature_vec, 
                        type_vec=type_vec,
                        group_vec=group_vec_list,
                        pred_vec=pred_adjusted, 
                        epoch=epoch)
            
            if epoch == 3:
                evaluate.BestBalance = 0.0
                evaluate.BestEntropy = 0.0
                evaluate.BestFairness = 0.0
                evaluate.BestNmiFair = 0.0
            
            # 更新进度条描述
            epoch_pbar.set_postfix({
                'NMI': f'{evaluate.BestNmiFair*100:.2f}%',
                'Balance': f'{evaluate.BestBalance*100:.2f}%',
                'Fairness': f'{evaluate.BestFairness*100:.2f}%'
            })
            
            # 训练阶段
            self.train()
            confidence_sum = 0.0
            loss_reconstruction_epoch = 0.0
            loss_reconstruct_all_epoch = 0.0
            discriminative_test_loss = 0.0
            loss_discriminative_epoch = 0.0
            loss_decenc_epoch = 0.0
            loss_global_balance_epoch = 0.0
            loss_info_global_epoch = 0.0
            loss_info_balance_epoch = 0.0
            loss_info_fair_epoch = 0.0
            loss_group_wise_balance_epoch = 0.0
            loss_cluster_wise_balance_epoch = 0.0
            loss_onehot_epoch = 0.0
            loss_consistency_epoch = 0.0
            loss_proto_epoch = 0.0
            loss_intersectional_balance_epoch = 0.0 # 新增

            # 使用tqdm创建批次进度条
            batch_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for i, (x, g, y, idx) in enumerate(batch_pbar):
                x = x.cuda()
                g = [g_i.cuda() for g_i in g]
                y = y.cuda()
                idx = idx.cuda()
                
                # 编码
                z = self.encode(x, g)
                
                # 解码（如只用一个敏感属性，明确指定g[1]，并加注释）
                x_recon = self.decode(z, g)  # 默认传g，内部如需只用g[1]已注明

                # 重构损失
                recon_loss = F.mse_loss(x_recon, x)
                
                # 判别器损失（如只用部分敏感属性，需明确指定）
                d_loss = torch.tensor(0.0, device=x.device)
                for i, g_i in enumerate(g):
                    d_loss = d_loss + self.discriminator_foreground[i % len(self.discriminator_foreground)](z[0].detach()).mean()
                
                # 平衡损失（遍历所有敏感属性）
                balance_loss = torch.tensor(0.0, device=x.device)
                for i, g_i in enumerate(g):
                    if args.GlobalBalanceLoss > 0:
                        balance_loss = balance_loss + args.GlobalBalanceLoss * self.group_wise_balance_loss(z[0], g)
                
                # 新增：交叉平衡损失
                intersectional_loss = torch.tensor(0.0, device=x.device)
                if args.LambdaIntersectionalBalance > 0:
                    probs = F.softmax(self.encode_class(z[0]), dim=1)
                    intersectional_loss = args.LambdaIntersectionalBalance * self.intersectional_balance_loss(probs, g)
                
                # 总损失
                loss = recon_loss + d_loss + balance_loss + intersectional_loss
                
                # 更新参数
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

                # 更新批次进度条
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'balance': f'{balance_loss.item():.4f}'
                })
                
                # 累积损失（遍历所有敏感属性，分别统计或融合）
                loss_reconstruction_epoch += recon_loss.item()
                loss_decenc_epoch += recon_loss.item()
                loss_discriminative_epoch += d_loss.item()
                loss_global_balance_epoch += balance_loss.item()
                if intersectional_loss > 0:
                    loss_intersectional_balance_epoch += intersectional_loss.item()
                # 下面这些损失，遍历所有敏感属性求平均
                loss_info_global_epoch += self.group_wise_balance_loss(z[0], g).item()
                loss_info_balance_epoch += self.group_wise_balance_loss(z[0], g).item()
                loss_info_fair_epoch += self.cluster_wise_balance_loss(z[0], g).item()
                loss_group_wise_balance_epoch += self.group_wise_balance_loss(z[0], g).item()
                loss_cluster_wise_balance_epoch += self.cluster_wise_balance_loss(z[0], g).item()
                loss_onehot_epoch += self.onehot_loss(z[0], g[0]).item()  # onehot_loss只用第一个敏感属性
                loss_consistency_epoch += self.consistency_loss(z[0], g[0]).item()  # consistency_loss只用第一个敏感属性
                loss_proto_epoch += self.proto_loss(z[0], g[0]).item()  # proto_loss只用第一个敏感属性

            # 计算平均损失
            len_train_dataloader = len(train_dataloader)
            confidence_sum /= len_train_dataloader
            loss_reconstruction_epoch /= len_train_dataloader
            loss_reconstruct_all_epoch /= len_train_dataloader
            discriminative_test_loss /= len_train_dataloader
            loss_discriminative_epoch /= len_train_dataloader
            loss_decenc_epoch /= len_train_dataloader
            loss_global_balance_epoch /= len_train_dataloader
            loss_info_global_epoch /= len_train_dataloader
            loss_info_balance_epoch /= len_train_dataloader
            loss_info_fair_epoch /= len_train_dataloader
            loss_group_wise_balance_epoch /= len_train_dataloader
            loss_cluster_wise_balance_epoch /= len_train_dataloader
            loss_onehot_epoch /= len_train_dataloader
            loss_consistency_epoch /= len_train_dataloader
            loss_proto_epoch /= len_train_dataloader
            loss_intersectional_balance_epoch /= len_train_dataloader # 新增

            # 打印训练信息
            print('\nEpoch [{: 3d}/{: 3d}]'.format(epoch + 1, epochs), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_reconstruct_all_epoch != 0:
                print(', ReconstructAll:{:04f}'.format(loss_reconstruct_all_epoch), end='')
            if discriminative_test_loss != 0:
                print(', DiscriminativeTest:{:04f}'.format(discriminative_test_loss), end='')
            if loss_discriminative_epoch != 0:
                print(', Discriminative:{:04f}'.format(loss_discriminative_epoch), end='')
            if loss_decenc_epoch != 0:
                print(', Decenc:{:04f}'.format(loss_decenc_epoch), end='')
            if loss_global_balance_epoch != 0:
                print(', GlobalBalance:{:04f}'.format(loss_global_balance_epoch), end='')
            if loss_info_global_epoch != 0:
                print(', InfoGlobal:{:04f}'.format(loss_info_global_epoch), end='')
            if loss_info_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_info_balance_epoch), end='')
            if loss_info_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_info_fair_epoch), end='')
            if loss_group_wise_balance_epoch != 0:
                print(', GroupWiseBalance:{:04f}'.format(loss_group_wise_balance_epoch), end='')
            if loss_cluster_wise_balance_epoch != 0:
                print(', ClusterWiseBalance:{:04f}'.format(loss_cluster_wise_balance_epoch), end='')
            if loss_consistency_epoch != 0:
                print(', Consistency:{:04f}'.format(loss_consistency_epoch), end='')
            if loss_onehot_epoch != 0:
                print(', OneHot:{:04f}'.format(loss_onehot_epoch), end='')
            if loss_proto_epoch != 0:
                print(', Proto:{:04f}'.format(loss_proto_epoch), end='')
            if loss_intersectional_balance_epoch != 0: # 新增
                print(', IntersectionalBalance:{:04f}'.format(loss_intersectional_balance_epoch), end='')
            if confidence_sum != 0:
                print(', Confidence:{:04f}'.format(confidence_sum), end='')
            print()
            
            timer_train.time()
            timer_all.time()
            
            if (epoch + 1) % args.VisualFreq == 0:
                timer_train.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='Train')
                timer_infer.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='Infer')
                timer_all.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='All')
            
            if CodeTest:
                assert False


class MyNormalize(nn.Module):
    def __call__(self, x):
        return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))


class Gassilize(nn.Module):
    def __init__(self, args):
        if args.dataset == 'MTFL':
            warnings.warn('bad mean and std')
        super(Gassilize, self).__init__()

    def __call__(self, x):
        return (x - 0.39618438482284546) / 0.4320564270019531


class NetFCN(Net):
    def __init__(self, input_dim, class_num, group_nums, args, num_sensitive_attrs=2):
        super().__init__(class_num, group_nums, args=args, num_sensitive_attrs=num_sensitive_attrs)
        self.input_dim = input_dim
        self.encoder_out_dim = 512

        def get_encoder_list():
            if args.BatchNormType[0] == '1':
                return nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.encoder_out_dim),
                    nn.BatchNorm1d(self.encoder_out_dim),
                    nn.ReLU(),
                )
            elif args.BatchNormType[0] == '0':
                return nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.encoder_out_dim),
                    nn.ReLU(),
                )
            else:
                raise NotImplementedError('')

        def get_encoder_line_list():
            if args.BatchNormType[1] == '1':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
            elif args.BatchNormType[1] == '0':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
            else:
                raise NotImplementedError('')

        def get_decoder_list():
            m_list = [
                nn.Linear(1024, input_dim),
            ]
            if args.dataset == 'MTFL' or args.dataset == 'Office':
                if args.ActivationType == 'None':
                    pass
                elif args.ActivationType == 'Sigmoid':
                    m_list.append(nn.Sigmoid())
                elif args.ActivationType == 'Tanh':
                    m_list.append(nn.Tanh())
                elif args.ActivationType == 'Normalize':
                    m_list.append(MyNormalize())
                elif args.ActivationType == 'Gaussainlize':
                    m_list.append(Gassilize(args))
                elif args.ActivationType == 'GlS_GaussainlizeAndSigmoid':
                    m_list.append(Gassilize(args))
                    m_list.append(nn.Sigmoid())
                elif args.ActivationType == 'GlT_GaussainlizeAndTanh':
                    m_list.append(Gassilize(args))
                    m_list.append(nn.Tanh())
                else:
                    raise NotImplementedError('')
            else:
                raise NotImplementedError('')

            if args.BatchNormType[3] == '1':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim * 2 if args.WithFeatureB else self.encoder_out_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    *m_list
                )
            elif args.BatchNormType[3] == '0':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim * 2 if args.WithFeatureB else self.encoder_out_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    *m_list
                )
            else:
                raise NotImplementedError('')

        def get_decoder_line_list():
            if args.BatchNormType[2] == '1':
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.BatchNorm1d(self.encoder_out_dim),
                    nn.ReLU(),
                )
            elif args.BatchNormType[2] == '0':
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.ReLU(),
                )
            else:
                raise NotImplementedError('')

        # 为每个敏感属性创建独立的编码器
        self.encoder = nn.ModuleList([
            get_encoder_list() for _ in range(2 if args.GroupWiseEncoder else 1)  # 2个敏感属性
        ])
        
        self.encoder_linear = nn.ModuleList([
            get_encoder_line_list() for _ in range(2 if args.WithFeatureB else 1)
        ])
        
        self.decoder_linear = nn.ModuleList([
            get_decoder_line_list() for _ in range(2 if args.WithFeatureB else 1)
        ])
        
        # 为每个敏感属性创建独立的解码器
        self.decoder = nn.ModuleList([
            get_decoder_list() for _ in range(2 if args.GroupWiseDecoder else 1)  # 2个敏感属性
        ])
        
        # 为每个敏感属性创建独立的判别器
        self.discriminator_foreground = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(), 
                nn.Linear(self.representation_dim, 512),
                nn.BatchNorm1d(512), 
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256), 
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2), 
                nn.Linear(256, 1)
            )
            for _ in range(self.num_sensitive_attrs)
        ])
        self.initialize_weights(self.discriminator_foreground)

    def encode(self, x, group_indices):
        if self.args.GroupWiseEncoder:
            hh = torch.zeros((len(x), self.encoder_out_dim), device='cuda')
            g = group_indices[0]
            for g_val in torch.unique(g):
                # 将张量转换为整数
                g_val_int = int(g_val.item())
                ind = g_val == g
                hh[ind] = self.encoder[g_val_int](x[ind])
        else:
            hh = self.encoder[0](x)
        
        a = self.encoder_linear[0](hh)
        
        if self.args.RepresentationType == 'Normalize':
            a = F.normalize(a, dim=1)
        elif self.args.RepresentationType == 'None':
            pass
        elif self.args.RepresentationType == 'Relu':
            pass
        else:
            raise NotImplementedError('RepresentationType')
        
        if self.args.WithFeatureB:
            b = self.encoder_linear[1](hh)
            if self.args.RepresentationType == 'Normalize':
                b = F.normalize(b, dim=1)
            elif self.args.RepresentationType == 'None':
                pass
            elif self.args.RepresentationType == 'Relu':
                b = nn.ReLU()(b)
            else:
                raise NotImplementedError('RepresentationType')
        else:
            b = None
            
        return a, b

    def decode(self, z, group_indices):
        if self.args.WithFeatureB:
            z = torch.cat([dec(zi) for zi, dec in zip(z, self.decoder_linear)], dim=1)
        else:
            z = self.decoder_linear[0](z[0])
        
        if self.args.GroupWiseDecoder:
            x_ = torch.zeros((len(z), self.input_dim), device='cuda')
            g = group_indices[1]  # 使用第二个敏感属性
            for g_val in torch.unique(g):
                ind = g_val == g
                if torch.sum(ind) == 0:
                    pass
                elif torch.sum(ind) == 1:
                    x_[ind] = self.decoder[g_val](torch.cat([z[ind], z[ind]], dim=1))[[0]]
                else:
                    x_[ind] = self.decoder[g_val](z[ind])
        else:
            x_ = self.decoder[0](z)
        return x_


def test():
    def get_kl_loss(type_num, group_num, n, batch_num=1000):
        mse_loss = nn.MSELoss().cuda()

        def get_loss(matrix_a, matrix_b):
            # if args.BalanceLossType == 'KL':
            # return F.kl_div(torch.log(matrix_a), matrix_b.detach(), reduction="batchmean") * (n / 20000) / (group_num / 2)
            # elif args.BalanceLossType == 'MSE':
            return mse_loss(matrix_a, matrix_b) * (n / 20000) * (group_num / 2) * (type_num / 10) ** 2
            # elif args.BalanceLossType == 'MAE':
            #     pass
            # else:
            #     raise NotImplementedError('BalanceLossType')

        loss = 0

        for _ in range(batch_num):
            # c_ = torch.rand((n, type_num))*2-1
            # c_ /= torch.sum(c_, dim=1, keepdim=True)
            c_ = torch.softmax(torch.cos(torch.rand((n, type_num)) * torch.pi) / 0.1, dim=1)
            # print(c_[:2])
            g = torch.as_tensor(torch.floor(torch.rand(n) * group_num), dtype=torch.int)
            # print(g)
            O = torch.zeros((type_num, group_num)).cuda()
            E = torch.zeros((type_num, group_num)).cuda()
            for b in range(group_num):
                O[:, b] = torch.sum(c_[g == b], dim=0)
                E[:, b] = (g == b).sum()

            # if args.GlobalBalance:
            loss += get_loss(
                (O / torch.sum(O)).view((1, -1)),
                (E / torch.sum(E)).view((1, -1))
            )
        # if args.GroupWiseBalanceLoss:
        #     loss +=  get_loss(
        #         (O / torch.sum(O, dim=1, keepdim=True)),
        #         (E / torch.sum(E, dim=1, keepdim=True))
        #     ) * n / group_num
        #     loss_group_wise_balance_epoch += group_wise_balance_loss.item()
        #     loss += group_wise_balance_loss * args.GroupWiseBalanceLoss
        # if args.ClusterWiseBalanceLoss:
        #     loss += get_loss(
        #         (O / torch.sum(O, dim=0, keepdim=True)).transpose(1, 0),
        #         (E / torch.sum(E, dim=0, keepdim=True)).transpose(1, 0)
        #     ) * n / group_num
        #     loss_cluster_wise_balance_epoch += cluster_wise_balance_loss.item()
        #     loss += cluster_wise_balance_loss * args.ClusterWiseBalanceLoss
        #
        # matrix_a = torch.rand((4, 3))
        # matrix_b = torch.ones((4, 3))
        # matrix_a /= torch.sum(matrix_a, dim=1, keepdim=True)
        # matrix_b /= torch.sum(matrix_b, dim=1, keepdim=True)
        # kl = F.kl_div(torch.log(matrix_a), matrix_b.detach(), reduction="batchmean")
        print('{:.08f}'.format(loss / batch_num))

    get_kl_loss(type_num=10, group_num=2, n=512)
    get_kl_loss(type_num=10, group_num=2, n=512)
    # get_kl_loss(type_num=10, group_num=2, n=5120)
    # get_kl_loss(type_num=10, group_num=30, n=5120)
    get_kl_loss(type_num=100, group_num=2, n=512)


# 0.16537261 0.08160374
# 0.16192900 0.08480503
# 0.16470969 0.08299696
# 0.17563063 0.20080499
# 0.16742396 0.08369189
def test2():
    type_num = 10
    group_num = 2
    n = 512
    O = torch.zeros((type_num, group_num))
    E = torch.zeros((type_num, group_num))
    c_ = torch.softmax(torch.cos(torch.rand((n, type_num)) * torch.pi) / 0.1, dim=1)
    # print(c_[:2])
    g = torch.as_tensor(torch.floor(torch.rand(n) * group_num), dtype=torch.int)
    print(c_)
    for b in range(group_num):
        O[:, b] = torch.sum(c_[g == b], dim=0)
        E[:, b] = (g == b).sum()
    print(O)


if __name__ == '__main__':
    test()
    # matrix_a = torch.linspace(1, 11, 12).view((4, -1))
    # matrix_b = torch.linspace(1, 111, 12).view((4, -1))

    # my_kl = torch.mean(torch.sum(matrix_a * torch.log(matrix_a / matrix_b), dim=1))
    # print(my_kl)

    # my_kl2 = torch.mean(torch.sum(matrix_b * torch.log(matrix_b / matrix_a), dim=1))
    # print(my_kl2)
