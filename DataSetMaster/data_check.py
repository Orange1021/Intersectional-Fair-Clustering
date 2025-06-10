import os
import re
# from compute_dist import Split
from evaluate import evaluate, evaluate2
import numpy as np
import torch
import dataset
from Utils.DirectoryOperator import FileOperator

Split = 'MTFL'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##验证聚类/分类模型在测试集上的性能表现，按不同组别（`group`）分别评估。
def check():
    _, test_dataloader, _ = dataset.get_dataloader(Split)
    feature_vec, type_vec, group_vec, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, g, y, idx) in test_dataloader:
            x, g, y = x.to(device), g.to(device), y.to(device)
            # pred = torch.argmax(c, dim=1)
            # feature_vec.extend(h[0].detach().cpu().numpy())
            type_vec.extend(y.cpu().numpy())
            group_vec.extend(g.cpu().numpy())
            # pred_vec.extend(pred.cpu().numpy())
    feature_vec, type_vec, group_vec, pred_vec = (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(group_vec),
        np.array(pred_vec),
    )

    class It:
        def __init__(self):
            self.dataset = Split

    pred_vec = dataset.get_clusters(It()).cpu().numpy()

    for y in np.unique(group_vec):
        # print(type_vec[group_vec == y])
        # print(pred_vec[np.sort(group_vec) == y])
        evaluate(feature_vec=None, pred_vec=pred_vec[group_vec == y], type_vec=type_vec[group_vec == y],
                 group_vec=None, best_acc=0)
    # assert group_vec[i]<=group_vec[j] for i<j
    evaluate(feature_vec=None, pred_vec=pred_vec, type_vec=type_vec, group_vec=None, best_acc=0)


##调整预测结果文件以优化公平性指标，并保存调整后的结果。
def rewrite():
    _, test_dataloader, _ = dataset.get_dataloader(Split)
    feature_vec, type_vec, group_vec, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, g, y, idx) in test_dataloader:
            # pred = torch.argmax(c, dim=1)
            # feature_vec.extend(h[0].detach().cpu().numpy())
            type_vec.extend(y.cpu().numpy())
            group_vec.extend(g.cpu().numpy())
            # pred_vec.extend(pred.cpu().numpy())
    feature_vec, type_vec, group_vec, pred_vec = (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(group_vec),
        np.array(pred_vec),
    )
    if Split != 'HAR':
        pth = '/root/autodl-tmp/Checkpoints/FairClustering/0314/ItemsNoAdjust/webcamOurLoaderRevalveBatchWiseB032_84.03.txt'
        pred_vec = np.asarray(np.loadtxt(pth), dtype=int)
        pred_adjusted = evaluate2(feature_vec=None, pred_vec=pred_vec, type_vec=type_vec, group_vec=None)
        evaluate2(feature_vec=None, pred_vec=pred_adjusted, type_vec=type_vec, group_vec=None)
        np.savetxt(pth.replace('NoAdjust', ''),
                   np.array(pred_adjusted).astype(np.int32),
                   fmt='%d')
    else:
        class It:
            def __init__(self):
                self.dataset = 'HAR'

        pred_vec = dataset.get_clusters(It()).cpu().numpy()
        # print(pred_vec)
        # print(np.max(pred_vec))

        ind = np.argsort(group_vec)
        group_vec = group_vec[ind]
        type_vec = type_vec[ind]
        for y in np.unique(group_vec):
            pred_adjusted = evaluate2(feature_vec=None, pred_vec=pred_vec[group_vec == y],
                                      type_vec=type_vec[group_vec == y], group_vec=None)
            evaluate2(feature_vec=None, pred_vec=pred_adjusted, type_vec=type_vec[group_vec == y], group_vec=None)
            # np.savetxt('/xlearning/pengxin/Checkpoints/FairClustering/0314/Items/har_pre/cluster_group_{:02d}.txt'.format(y),
            #            np.array(pred_adjusted).astype(np.int32),
            #            fmt='%d')



##从多个实验结果文件中筛选每组（`group`）准确率最高的文件，复制到目标目录并重命名。
def file_clean():
    src_root = '/autodl-tmp/Checkpoints/FairClustering/MTFL'
    dst_root = '/autodl-tmp/Checkpoints/FairClustering/Items0321/MTFL'
    g_acc = {}
    g_name = {}
    for root, folds, files in os.walk(src_root, topdown=False):
        for file in files:
            if file.endswith('txt'):
                acc = float(file[-9:-4])
                g = int(file[-12:-10])
                if g not in g_acc or g_acc[g] < acc:
                    g_acc[g] = acc
                    g_name[g] = os.path.join(root, file)
    for g, name in g_name.items():
        # print(g, name)
        res = re.findall(r'B\d\d\d', name)[0]
        result = os.path.basename(name).replace(res, '').replace('.txt', res + '.txt')
        # print(g, name, result)

        FileOperator(name).copy(os.path.join(dst_root, result))


if __name__ == "__main__":
    check()
