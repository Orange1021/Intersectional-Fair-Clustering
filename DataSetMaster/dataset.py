import os
import warnings
import numpy as np
import torch
from torch.utils import data
import torchvision
from PIL import Image

from MainLauncher import path_operator
from DataSetMaster.sample_utils import SamplingDataset, MultiSensitiveSamplingDataset

# from Utils import PathPresettingOperator
#
# path_operator = PathPresettingOperator.PathOperator(model_name='FairClustering')

# 通用敏感属性增强工具
from torch.utils.data import Dataset
class CustomSensitiveDataset(Dataset):
    def __init__(self, base_dataset, attr_specs):
        self.base_dataset = base_dataset
        self.length = len(base_dataset)
        self.num_attrs = len(attr_specs)
        self.sensitive_attrs = []
        np.random.seed(42)  # 设置随机种子以保证可复现性
        for spec in attr_specs:
            num_classes = spec['num_classes']
            ratios = spec['ratios']
            assert abs(sum(ratios) - 1.0) < 1e-6, "ratios之和必须为1"
            arr = []
            for cls, ratio in enumerate(ratios):
                arr += [cls] * int(self.length * ratio)
            # 补齐长度
            while len(arr) < self.length:
                arr.append(num_classes - 1)
            arr = arr[:self.length]
            np.random.shuffle(arr)
            self.sensitive_attrs.append(arr)
        self.sensitive_attrs = np.array(self.sensitive_attrs).T.tolist()
    def __getitem__(self, idx):
        img, g_list, target, index = self.base_dataset[idx]
        if isinstance(g_list, list):
            new_g_list = g_list + self.sensitive_attrs[idx]
        else:
            new_g_list = [g_list] + self.sensitive_attrs[idx]
        return img, new_g_list, target, index
    def __len__(self):
        return self.length

NumWorkers = 4


##加载指定数据集，支持多种类型（如 MNIST、Office、HAR 等），并返回处理后的 `Dataset` 对象及类别数量。
def get_dataset(dataset, group=-1, path=path_operator.get_dataset_path, args=None, res_selector=None, transforms=None,
                flatten=False, sampling_method=None, train_ratio=0.8, attr_specs=None, feature_extractor='resnet50'):
    """
    获取数据集，支持采样方法，返回训练集和测试集
    :param dataset: 数据集名称
    :param group: 组别
    :param path: 数据路径
    :param args: 参数
    :param res_selector: 结果选择器
    :param transforms: 数据转换
    :param flatten: 是否展平
    :param sampling_method: 采样方法，可选值：None(不使用采样), 'bm'(bias-mimicking), 'us'(under-sampling), 'os'(over-sampling)
    :param train_ratio: 训练集比例，默认0.8
    :param attr_specs: 人工敏感属性配置（list），每个元素为{'num_classes': int, 'ratios': list}
    :param feature_extractor: 特征提取器，可选 'resnet50' 或 'dinov2'
    :return: (train_dataset, test_dataset), class_num, feature_dim
    """
    if args is None:
        mnist_train = False
    else:
        mnist_train = args.MnistTrain

    class MyImageFolder_0(torchvision.datasets.ImageFolder):
        def __init__(self, root, transform, *args, **kwargs):
            super(MyImageFolder_0, self).__init__(root, transform, *args, **kwargs)
            # 使用固定的随机种子生成敏感属性
            torch.manual_seed(42)  # 设置随机种子
            self.sensitive_attrs = torch.randint(0, 2, (len(self.samples),))
            
        def __getitem__(self, index: int):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # 计算图片亮度
            if isinstance(sample, torch.Tensor):
                brightness = torch.mean(sample)
                brightness_group = 1 if brightness > 0.5 else 0
            else:
                gray_img = sample.convert('L')
                brightness = np.mean(np.array(gray_img)) / 255.0
                brightness_group = 1 if brightness > 0.5 else 0
            
            # 返回两个敏感属性：数据源(0)和随机生成的敏感属性
            g_list = [0, self.sensitive_attrs[index]]
            img, g_list, target, index = sample, g_list, target, index
            return img, g_list, target, index

    class MyImageFolder_1(torchvision.datasets.ImageFolder):
        def __init__(self, root, transform, ind_bias=2817, *args, **kwargs):
            super(MyImageFolder_1, self).__init__(root, transform, *args, **kwargs)
            self.ind_bias = ind_bias
            # 使用固定的随机种子生成敏感属性
            torch.manual_seed(42)  # 使用统一的种子
            self.sensitive_attrs = torch.randint(0, 2, (len(self.samples),))
            
        def __getitem__(self, index: int):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # 计算图片亮度
            if isinstance(sample, torch.Tensor):
                brightness = torch.mean(sample)
                brightness_group = 1 if brightness > 0.5 else 0
            else:
                gray_img = sample.convert('L')
                brightness = np.mean(np.array(gray_img)) / 255.0
                brightness_group = 1 if brightness > 0.5 else 0
            
            # 返回两个敏感属性：数据源(1)和随机生成的敏感属性
            g_list = [1, self.sensitive_attrs[index]]
            img, g_list, target, index = sample, g_list, target, index + self.ind_bias
            return img, g_list, target, index

    class MNIST(torchvision.datasets.MNIST):
        def __init__(self, *args, **kwargs):
            super(MNIST, self).__init__(*args, **kwargs)
            # 使用固定的随机种子生成敏感属性
            torch.manual_seed(42)  # 设置随机种子
            # 生成 6:4 比例的敏感属性
            total_samples = len(self.targets)
            num_group_0 = int(total_samples * 0.6)  # 60% 的样本为组 0
            num_group_1 = total_samples - num_group_0  # 40% 的样本为组 1
            self.sensitive_attrs = torch.cat([
                torch.zeros(num_group_0),
                torch.ones(num_group_1)
            ])
            # 随机打乱顺序
            self.sensitive_attrs = self.sensitive_attrs[torch.randperm(total_samples)]
            
        def __getitem__(self, index: int):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # 使用随机分配的敏感属性
            g = self.sensitive_attrs[index]
            
            if flatten:
                img = img.view((-1))
            if res_selector is not None:
                return np.asarray([img, g, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class USPS(torchvision.datasets.USPS):
        def __init__(self, *arg, **kwargs):
            warnings.warn('USPS index')
            super(USPS, self).__init__(*arg, **kwargs)
            # 使用固定的随机种子生成敏感属性
            torch.manual_seed(42)  # 使用统一的种子
            # 生成 6:4 比例的敏感属性
            total_samples = len(self.targets)
            num_group_0 = int(total_samples * 0.6)  # 60% 的样本为组 0
            num_group_1 = total_samples - num_group_0  # 40% 的样本为组 1
            self.sensitive_attrs = torch.cat([
                torch.zeros(num_group_0),
                torch.ones(num_group_1)
            ])
            # 随机打乱顺序
            self.sensitive_attrs = self.sensitive_attrs[torch.randperm(total_samples)]
            
        def __getitem__(self, index: int):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # 使用随机分配的敏感属性
            g = self.sensitive_attrs[index]

            if flatten:
                img = img.view((-1))
            img, g, target, index = img, g, target, index + (60000 if mnist_train else 10000)
            if res_selector is not None:
                return np.asarray([img, g, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class Rev_MNIST(torchvision.datasets.MNIST):
        def __init__(self, *arg, **kwargs):
            warnings.warn('Rev_MNIST index')
            super(Rev_MNIST, self).__init__(*arg, **kwargs)
            # 使用固定的随机种子生成敏感属性
            torch.manual_seed(42)  # 使用统一的种子
            # 生成 6:4 比例的敏感属性
            total_samples = len(self.targets)
            num_group_0 = int(total_samples * 0.6)  # 60% 的样本为组 0
            num_group_1 = total_samples - num_group_0  # 40% 的样本为组 1
            self.sensitive_attrs = torch.cat([
                torch.zeros(num_group_0),
                torch.ones(num_group_1)
            ])
            # 随机打乱顺序
            self.sensitive_attrs = self.sensitive_attrs[torch.randperm(total_samples)]
            
        def __getitem__(self, index: int):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # 使用随机分配的敏感属性
            g = self.sensitive_attrs[index]
            
            if flatten:
                img = img.view((-1))
            img, g, target, index = img, g, target, index + (60000 if mnist_train else 10000)
            if res_selector is not None:
                return np.asarray([img, g, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class SensitiveTensorDataset(data.Dataset):
        def __init__(self, img, g_list, target, index):
            self.img = img
            self.g_list = g_list  # 敏感属性列表
            self.target = target
            self.index = index
            
        def __getitem__(self, idx):
            return self.img[idx], [g[idx] for g in self.g_list], self.target[idx], self.index[idx]
        
        def __len__(self):
            return len(self.img)

    class MultiSensitiveDataset(data.Dataset):
        def __init__(self, dataset, source_group=0):
            self.dataset = dataset
            self.source_group = source_group  # 0 for MNIST, 1 for USPS/Rev_MNIST
            
        def __getitem__(self, index):
            img, g, target, idx = self.dataset[index]
            # 数据源作为第二个敏感属性
            g_list = [g, self.source_group]
            return img, g_list, target, idx
            
        def __len__(self):
            return len(self.dataset)

    def featurelize(dataset, resnet_weight_path=None, device='cuda', feature_extractor='resnet50', args=None):
        test_loader = data.DataLoader(dataset, batch_size=512, num_workers=NumWorkers)
        
        feature_dim = 0
        if feature_extractor == 'resnet50':
            print("使用 ResNet50 提取特征...")
            if resnet_weight_path is not None:
                net = get_resnet50_from_pretrain(resnet_weight_path, device=device)
            else:
                net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device)
            feature_dim = 2048
        elif feature_extractor == 'dinov2':
            print("使用 DINOv2 提取特征...")
            # 从本地代码库加载模型结构，然后加载指定的权重文件。
            dinov2_repo_path = '/root/FCMI/PreTrain/dinov2-main'
            dinov2_weights_path = '/root/FCMI/PreTrain/dinov2_vitg14_pretrain.pth'
            
            # 1. 加载模型结构 (通过 pretrained=False 阻止自动下载)
            net = torch.hub.load(dinov2_repo_path, 'dinov2_vitg14', source='local', pretrained=False).to(device)
            
            # 2. 从指定路径加载权重
            state_dict = torch.load(dinov2_weights_path, map_location=device)
            
            # 3. 将权重载入模型
            net.load_state_dict(state_dict)
            
            feature_dim = 1536
        else:
            raise ValueError(f"不支持的特征提取器: {feature_extractor}")

        net.eval()

        # newnorm
        if args is not None and hasattr(args, 'dataset'):
            if args.dataset =='MTFL':
                mean = 0.34949973225593567
                std = 0.3805956244468689
            elif args.dataset =='Office':
                mean = 0.3970963954925537
                std = 0.43060600757598877
            else:
                mean = None
                std = None
        else:
            mean = None
            std = None

        def forward(net, x):
            if feature_extractor == 'resnet50':
                x = net.conv1(x)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                x = net.layer4(x)
                x = net.avgpool(x)
                x = torch.flatten(x, 1)
            elif feature_extractor == 'dinov2':
                x = net(x) # DINOv2直接调用即可
            
            if args is None or args.FeatureType == 'Tanh':
                x = torch.nn.Tanh()(x)
            elif args.FeatureType == 'Gaussainlize' and mean is not None and std is not None:
                x = (x - mean) / std
            elif args.FeatureType == 'Normalize':
                x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
            elif args.FeatureType == 'Default':
                pass
            elif args.FeatureType == 'GlS_GaussainlizeAndSigmoid' and mean is not None and std is not None:
                x = torch.nn.Sigmoid()((x - mean) / std)
            elif args.FeatureType == 'GlT_GaussainlizeAndTanh' and mean is not None and std is not None:
                x = torch.nn.Tanh()((x - mean) / std)
            elif args.FeatureType == 'Sigmoid':
                x = torch.nn.Sigmoid()(x)
            else:
                raise NotImplementedError('FeatureType')
            return x

        net.eval()
        feature_vec, type_vec, group_vec_list, idx_vec = [], [], [], []
        with torch.no_grad():
            for (x, g_list, y, idx) in test_loader:
                x = x.to(device)
                g_list = [g_i.to(device) for g_i in g_list]
                c = forward(net, x)

                feature_vec.extend(c.cpu().numpy())
                type_vec.extend(y.cpu().numpy())
                if len(group_vec_list) == 0:
                    group_vec_list = [[] for _ in range(len(g_list))]
                for i, g_i in enumerate(g_list):
                    if len(g_i.shape) > 1:
                        g_i = g_i.squeeze()
                    group_vec_list[i].extend(g_i.cpu().numpy())
                idx_vec.extend(idx.cpu().numpy())
        x = torch.from_numpy(np.array(feature_vec))
        img = x
        g_tensors = [torch.tensor(g, dtype=torch.long) for g in group_vec_list]
        target = torch.tensor(type_vec, dtype=torch.long)
        index = torch.tensor(idx_vec, dtype=torch.long)
        if flatten:
            img = img.view((len(img), -1))
        return SensitiveTensorDataset(img, g_tensors, target, index), feature_dim

    if dataset == 'Office':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.7076, 0.7034, 0.7021],
                                               std=[0.3249, 0.3261, 0.3275]),
            ])
        
        print("\n加载 Office 数据集...")
        # 加载Amazon和Webcam数据集
        amazon = MyImageFolder_0('/root/autodl-tmp/Data/Office/amazon/images',
                               transform=transforms)
        webcam = MyImageFolder_1('/root/autodl-tmp/Data/Office/webcam/images',
                               transform=transforms)
        
        # ====== 敏感属性增强：添加2个敏感属性（与原有2个合计4个） ======
        if attr_specs is None:
            attr_specs = [
                {'num_classes': 2, 'ratios': [0.1, 0.9]},
                {'num_classes': 3, 'ratios': [0.2, 0.7, 0.1]}
            ]
        amazon = CustomSensitiveDataset(amazon, attr_specs)
        webcam = CustomSensitiveDataset(webcam, attr_specs)
        
        print(f"Amazon 数据集大小: {len(amazon)}")
        print(f"Webcam 数据集大小: {len(webcam)}")
        
        # 计算每个数据集的训练集大小
        amazon_size = len(amazon)
        webcam_size = len(webcam)
        amazon_train_size = int(amazon_size * train_ratio)
        webcam_train_size = int(webcam_size * train_ratio)
        
        print(f"训练集划分比例: {train_ratio}")
        print(f"Amazon 训练集大小: {amazon_train_size}")
        print(f"Webcam 训练集大小: {webcam_train_size}")
        
        # 使用固定的随机种子进行划分
        torch.manual_seed(42)
        amazon_indices = torch.randperm(amazon_size)
        webcam_indices = torch.randperm(webcam_size)
        
        # 划分训练集和测试集
        amazon_train_indices = amazon_indices[:amazon_train_size]
        amazon_test_indices = amazon_indices[amazon_train_size:]
        webcam_train_indices = webcam_indices[:webcam_train_size]
        webcam_test_indices = webcam_indices[webcam_train_size:]
        
        # 创建训练集和测试集
        if group == -1:
            train_dataset = data.ConcatDataset([
                data.Subset(amazon, amazon_train_indices),
                data.Subset(webcam, webcam_train_indices)
            ])
            test_dataset = data.ConcatDataset([
                data.Subset(amazon, amazon_test_indices),
                data.Subset(webcam, webcam_test_indices)
            ])
        elif group == 0:
            train_dataset = data.Subset(amazon, amazon_train_indices)
            test_dataset = data.Subset(amazon, amazon_test_indices)
        elif group == 1:
            train_dataset = data.Subset(webcam, webcam_train_indices)
            test_dataset = data.Subset(webcam, webcam_test_indices)
        else:
            raise NotImplementedError('group')
        
        print(f"合并后训练集大小: {len(train_dataset)}")
        print(f"合并后测试集大小: {len(test_dataset)}")
        
        print(f"\n使用 {feature_extractor} 提取特征...")
        # 使用featurelize处理数据集
        train_dataset, feature_dim = featurelize(dataset=train_dataset, feature_extractor=feature_extractor, args=args)
        test_dataset, _ = featurelize(dataset=test_dataset, feature_extractor=feature_extractor, args=args)
        
        print(f"特征提取后训练集大小: {len(train_dataset)}")
        print(f"特征提取后测试集大小: {len(test_dataset)}")
        
        # 检查敏感属性的数量
        sample_data = train_dataset[0]
        num_sensitive_attrs = len(sample_data[1]) if isinstance(sample_data[1], list) else 1
        
        # 根据敏感属性数量选择采样数据集类
        if num_sensitive_attrs <= 3:
            print("使用标准采样数据集类（敏感属性数量 <= 3）")
            train_dataset = SamplingDataset(train_dataset)
        else:
            print("使用多敏感属性采样数据集类（敏感属性数量 > 3）")
            train_dataset = MultiSensitiveSamplingDataset(train_dataset)

        # 仅在非多敏感属性采样时应用常规采样方法
        if not isinstance(train_dataset, MultiSensitiveSamplingDataset):
            if sampling_method == 'bm':
                print("执行 bias-mimicking 采样...")
                train_dataset.bias_mimick()
            elif sampling_method == 'us':
                print("执行 under-sampling 采样...")
                train_dataset.under_sample_ce()
            elif sampling_method == 'os':
                print("执行 over-sampling 采样...")
                train_dataset.over_sample_ce()
            else:
                raise ValueError(f"不支持的采样方法: {sampling_method}")
        
            # 打印采样后的分布情况
            print("\n采样后的分布情况:")
            train_dataset.print_new_distro()
        
        class_num = 31
        return (train_dataset, test_dataset), class_num, feature_dim
    elif dataset == 'MNISTUSPS':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        
        # 加载 MNIST 数据集
        mnist_train = MultiSensitiveDataset(
            MNIST(train=True,
                  download=True,
                  root=path(''),
                  transform=transforms),
            source_group=0
        )
        mnist_test = MultiSensitiveDataset(
            MNIST(train=False,
                  download=True,
                  root=path(''),
                  transform=transforms),
            source_group=0
        )
        
        # 加载 USPS 数据集
        usps_train = MultiSensitiveDataset(
            USPS(train=True,
                 download=True,
                 root='/root/autodl-tmp/Data/USPS',
                 transform=transforms),
            source_group=1
        )
        usps_test = MultiSensitiveDataset(
            USPS(train=False,
                 download=True,
                 root='/root/autodl-tmp/Data/USPS',
                 transform=transforms),
            source_group=1
        )

        # # ====== 敏感属性增强：添加2个敏感属性（与原有2个合计4个） ======
        # if attr_specs is None:
        #     attr_specs = [
        #         {'num_classes': 2, 'ratios': [0.3, 0.7]},
        #         {'num_classes': 3, 'ratios': [0.3, 0.3, 0.4]}
        #     ]
        # mnist_train = CustomSensitiveDataset(mnist_train, attr_specs)
        # mnist_test = CustomSensitiveDataset(mnist_test, attr_specs)
        # usps_train = CustomSensitiveDataset(usps_train, attr_specs)
        # usps_test = CustomSensitiveDataset(usps_test, attr_specs)
        
        # 根据 group 参数选择数据集
        if group == -1:
            train_dataset = data.ConcatDataset([mnist_train, usps_train])
            test_dataset = data.ConcatDataset([mnist_test, usps_test])
        elif group == 0:
            train_dataset = mnist_train
            test_dataset = mnist_test
        elif group == 1:
            train_dataset = usps_train
            test_dataset = usps_test
        else:
            raise NotImplementedError('group')
            
        # 检查敏感属性的数量
        sample_data = train_dataset[0]
        num_sensitive_attrs = len(sample_data[1]) if isinstance(sample_data[1], list) else 1
        
        # 根据敏感属性数量选择采样数据集类
        if num_sensitive_attrs <= 3:
            print("使用标准采样数据集类（敏感属性数量 <= 3）")
            train_dataset = SamplingDataset(train_dataset)
        else:
            print("使用多敏感属性采样数据集类（敏感属性数量 > 3）")
            train_dataset = MultiSensitiveSamplingDataset(train_dataset)

        # 仅在非多敏感属性采样时应用常规采样方法
        if not isinstance(train_dataset, MultiSensitiveSamplingDataset):
            if sampling_method == 'bm':
                print("执行 bias-mimicking 采样...")
                train_dataset.bias_mimick()
            elif sampling_method == 'us':
                print("执行 under-sampling 采样...")
                train_dataset.under_sample_ce()
            elif sampling_method == 'os':
                print("执行 over-sampling 采样...")
                train_dataset.over_sample_ce()
            else:
                raise ValueError(f"不支持的采样方法: {sampling_method}")
        
            # 打印采样后的分布情况
            print("\n采样后的分布情况:")
            train_dataset.print_new_distro()
        
        class_num = 10
        return (train_dataset, test_dataset), class_num, 784 # MNIST/USPS 特征维度固定
    elif dataset == 'ReverseMNIST':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        
        # 创建反转图像的转换
        class Reverse:
            def __call__(self, img):
                return 1 - img
        
        rev_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
            Reverse(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        
        # 加载 MNIST 数据集
        mnist_train = MultiSensitiveDataset(
            MNIST(train=True,
                  download=True,
                  root=path(''),
                  transform=transforms),
            source_group=0
        )
        mnist_test = MultiSensitiveDataset(
            MNIST(train=False,
                  download=True,
                  root=path(''),
                  transform=transforms),
            source_group=0
        )
        
        # 加载 Reverse MNIST 数据集
        rev_mnist_train = MultiSensitiveDataset(
            Rev_MNIST(train=True,
                     download=True,
                     root=path(''),
                     transform=rev_transforms),
            source_group=1
        )
        rev_mnist_test = MultiSensitiveDataset(
            Rev_MNIST(train=False,
                     download=True,
                     root=path(''),
                     transform=rev_transforms),
            source_group=1
        )
        
        # 根据 group 参数选择数据集
        if group == -1:
            train_dataset = data.ConcatDataset([mnist_train, rev_mnist_train])
            test_dataset = data.ConcatDataset([mnist_test, rev_mnist_test])
        elif group == 0:
            train_dataset = mnist_train
            test_dataset = mnist_test
        elif group == 1:
            train_dataset = rev_mnist_train
            test_dataset = rev_mnist_test
        else:
            raise NotImplementedError('group')
            
        # 检查敏感属性的数量
        sample_data = train_dataset[0]
        num_sensitive_attrs = len(sample_data[1]) if isinstance(sample_data[1], list) else 1
        
        # 根据敏感属性数量选择采样数据集类
        if num_sensitive_attrs <= 3:
            print("使用标准采样数据集类（敏感属性数量 <= 3）")
            train_dataset = SamplingDataset(train_dataset)
        else:
            print("使用多敏感属性采样数据集类（敏感属性数量 > 3）")
            train_dataset = MultiSensitiveSamplingDataset(train_dataset)

        # 仅在非多敏感属性采样时应用常规采样方法
        if not isinstance(train_dataset, MultiSensitiveSamplingDataset):
            if sampling_method == 'bm':
                print("执行 bias-mimicking 采样...")
                train_dataset.bias_mimick()
            elif sampling_method == 'us':
                print("执行 under-sampling 采样...")
                train_dataset.under_sample_ce()
            elif sampling_method == 'os':
                print("执行 over-sampling 采样...")
                train_dataset.over_sample_ce()
            else:
                raise ValueError(f"不支持的采样方法: {sampling_method}")
        
            # 打印采样后的分布情况
            print("\n采样后的分布情况:")
            train_dataset.print_new_distro()
        
        class_num = 10
        return (train_dataset, test_dataset), class_num, 784 # MNIST/USPS 特征维度固定
    elif dataset == 'MTFL':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.4951, 0.4064, 0.3579],
                                               std=[0.2868, 0.2594, 0.2545]),
            ])
        
        class MTFLDataset(torchvision.datasets.ImageFolder):
            def __init__(self, root, transform, train=True, *args, **kwargs):
                super(MTFLDataset, self).__init__(root, transform, *args, **kwargs)
                self.train = train
                
                # 从路径中提取属性信息
                self.samples_info = []
                for path, _ in self.samples:
                    # 路径格式: .../g{0,1}/s{0,1}/y{0,1}/image.jpg
                    path_parts = path.split('/')
                    g = int(path_parts[-4][1])  # 眼镜属性 (g0/g1)
                    s = int(path_parts[-3][1])  # 微笑属性 (s0/s1)
                    y = int(path_parts[-2][1])  # 性别属性 (y0/y1)
                    self.samples_info.append((path, g, s, y))
                
                # 打印数据集信息
                print(f"\nMTFL {'训练' if train else '测试'}集信息:")
                total_samples = len(self.samples_info)
                print(f"总样本数: {total_samples}")
                
                # 统计属性分布
                gender_counts = {0: 0, 1: 0}  # 女性: 0, 男性: 1
                glasses_counts = {0: 0, 1: 0}  # 无眼镜: 0, 有眼镜: 1
                smile_counts = {0: 0, 1: 0}    # 不微笑: 0, 微笑: 1
                
                for _, g, s, y in self.samples_info:
                    gender_counts[y] += 1
                    glasses_counts[g] += 1
                    smile_counts[s] += 1
                
                print(f"性别分布 - 女性: {gender_counts[0]} ({gender_counts[0]/total_samples*100:.1f}%), "
                      f"男性: {gender_counts[1]} ({gender_counts[1]/total_samples*100:.1f}%)")
                print(f"眼镜分布 - 无眼镜: {glasses_counts[0]} ({glasses_counts[0]/total_samples*100:.1f}%), "
                      f"有眼镜: {glasses_counts[1]} ({glasses_counts[1]/total_samples*100:.1f}%)")
                print(f"微笑分布 - 不微笑: {smile_counts[0]} ({smile_counts[0]/total_samples*100:.1f}%), "
                      f"微笑: {smile_counts[1]} ({smile_counts[1]/total_samples*100:.1f}%)")
            
            def __getitem__(self, index: int):
                # 获取图像和属性信息
                path, g, s, y = self.samples_info[index]
                sample = self.loader(path)
                
                if self.transform is not None:
                    sample = self.transform(sample)
                
                # 将敏感属性转换为张量
                g_tensor = torch.tensor(g, dtype=torch.long)
                s_tensor = torch.tensor(s, dtype=torch.long)
                g_list = [g_tensor, s_tensor]  # 眼镜和微笑作为敏感属性
                
                return sample, g_list, y, index
        
        # 创建训练集和测试集
        train_dataset = MTFLDataset(
            root='/root/autodl-tmp/Data/MTFL/train',
            transform=transforms,
            train=True
        )
        
        test_dataset = MTFLDataset(
            root='/root/autodl-tmp/Data/MTFL/test',
            transform=transforms,
            train=False
        )
        
        # 使用 featurelize 处理数据集
        print(f"\n使用 {feature_extractor} 提取特征...")
        train_dataset, feature_dim = featurelize(dataset=train_dataset, feature_extractor=feature_extractor, args=args)
        test_dataset, _ = featurelize(dataset=test_dataset, feature_extractor=feature_extractor, args=args)
        
        # 检查敏感属性的数量
        sample_data = train_dataset[0]
        num_sensitive_attrs = len(sample_data[1]) if isinstance(sample_data[1], list) else 1
        
        # 根据敏感属性数量选择采样数据集类
        if num_sensitive_attrs <= 3:
            print("使用标准采样数据集类（敏感属性数量 <= 3）")
            train_dataset = SamplingDataset(train_dataset)
        else:
            print("使用多敏感属性采样数据集类（敏感属性数量 > 3）")
            train_dataset = MultiSensitiveSamplingDataset(train_dataset)

        # 仅在非多敏感属性采样时应用常规采样方法
        if not isinstance(train_dataset, MultiSensitiveSamplingDataset):
            if sampling_method == 'bm':
                print("执行 bias-mimicking 采样...")
                train_dataset.bias_mimick()
            elif sampling_method == 'us':
                print("执行 under-sampling 采样...")
                train_dataset.under_sample_ce()
            elif sampling_method == 'os':
                print("执行 over-sampling 采样...")
                train_dataset.over_sample_ce()
            else:
                raise ValueError(f"不支持的采样方法: {sampling_method}")
        
            # 打印采样后的分布情况
            print("\n采样后的分布情况:")
            train_dataset.print_new_distro()
        
        class_num = 2  # 性别分类（男/女）
        return (train_dataset, test_dataset), class_num, feature_dim
    else:
        raise NotImplementedError(f"数据集 {dataset} 尚未实现训练集和测试集的划分")


def get_dataloader(dataset, batch_size=512, sampling_method=None, **kwargs):
    """
    获取数据加载器，支持采样方法
    :param dataset: 数据集名称
    :param batch_size: 批次大小
    :param sampling_method: 采样方法，可选值：None(不使用采样), 'bm', 'us', 'os'
    :param kwargs: 其他参数
    :return: train_loader, test_loader, class_num, feature_dim
    """
    (train_dataset, test_dataset), class_num, feature_dim = get_dataset(dataset, sampling_method=sampling_method, **kwargs)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True, num_workers=NumWorkers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size * 100, num_workers=NumWorkers)

    return train_loader, test_loader, class_num, feature_dim


def get_clusters(args):
    item_path = os.path.join(path_operator.get_checkpoint_path(level=1), 'Items0321')
    file_mnist_test = os.path.join(item_path, 'mnist_test_clusters89.67.txt')
    file_mnist_train = os.path.join(item_path, 'MnistTrain94.31B256.txt')
    file_amazon = os.path.join(item_path, 'amazon72.81B032ReValue.txt')
    file_webcam = os.path.join(item_path, 'webcamOurLoaderRevalveBatchWiseB032_84.03.txt')
    file_usps = os.path.join(item_path, 'usps_train_clusters85.10.txt')
    root_har = os.path.join(item_path, 'HAR')
    root_mtfl = os.path.join(item_path, 'MTFL')

    if args.dataset == 'MNISTUSPS':  # 87.75 93.31
        if args.MnistTrain:
            file_mnist = file_mnist_train
        else:
            file_mnist = file_mnist_test
        file_list = [
            file_mnist,
            file_usps,
        ]
    elif args.dataset == 'ReverseMNIST':  # 89.67 94.31
        if args.MnistTrain:
            file_mnist = file_mnist_train
        else:
            file_mnist = file_mnist_test
        file_list = [
            file_mnist,
            file_mnist,
        ]
    elif args.dataset == 'Office':  # 75.28
        file_list = [
            file_amazon,
            file_webcam,
        ]
    elif args.dataset == 'MTFL':
        file_list = np.sort([os.path.join(root_mtfl, f) for f in os.listdir(root_mtfl) if f.endswith('txt')])
    elif args.dataset == 'HAR':  # 81.70
        file_list = np.sort([os.path.join(root_har, f) for f in os.listdir(root_har) if f.endswith('txt')])
    else:
        raise NotImplementedError("NotImplementedError")

    def debug(x):
        print(x.shape)
        return x

    clusters = torch.cat(
        [debug(torch.from_numpy(np.loadtxt(c).astype(np.float32)).long()) for c in file_list],
        dim=0,
    ).cuda()
    return clusters

# ==== 以下为SimCLR自监督预训练ResNet50相关函数 ====
def pretrain_resnet_simclr(img_dir, epochs=100, batch_size=128, lr=1e-3, out_dim=128, device='cuda', save_path='simclr_resnet50.pth'):
    """
    用SimCLR自监督方法预训练ResNet50主干网络，并保存权重。
    依赖 lightly 库。
    """
    try:
        from lightly.models.modules.heads import SimCLRProjectionHead
        from lightly.loss import NTXentLoss
        from lightly.data import LightlyDataset
        from lightly.data.collate import SimCLRCollateFunction
        from lightly.models import SimCLR
    except ImportError:
        raise ImportError('请先 pip install lightly')

    # 数据增强
    collate_fn = SimCLRCollateFunction(input_size=224, cj_strength=0.5)
    dataset = LightlyDataset(input_dir=img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    # 模型
    import torch.nn as nn
    backbone = torchvision.models.resnet50(weights=None)
    backbone.fc = nn.Identity()  # 去掉最后的全连接层，输出2048维
    model = SimCLR(backbone, num_ftrs=2048, out_dim=out_dim)
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    # 训练
    from tqdm import tqdm
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_iter = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for (x0, x1), _, _ in batch_iter:
            x0, x1 = x0.to(device), x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_iter.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'[SimCLR] Epoch {epoch+1}/{epochs}, avg_loss={total_loss/len(dataloader):.4f}')
    # 保存主干网络
    torch.save(model.backbone.state_dict(), save_path)
    print(f'[SimCLR] 预训练ResNet50已保存到: {save_path}')
    return save_path

def get_resnet50_from_pretrain(weight_path, device='cuda'):
    """
    加载预训练权重的ResNet50主干网络。
    """
    backbone = torchvision.models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()  # 保证结构一致
    state_dict = torch.load(weight_path, map_location=device)
    backbone.load_state_dict(state_dict, strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone

def extract_features_with_resnet(dataset, backbone, device='cuda'):
    """
    用预训练ResNet50主干提取特征，返回特征张量。
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    features = []
    with torch.no_grad():
        for (img, *_) in dataloader:
            img = img.to(device)
            feat = backbone(img)
            features.append(feat.cpu())
    features = torch.cat(features, dim=0)
    print(f'[Feature] 提取特征 shape: {features.shape}')
    return features
