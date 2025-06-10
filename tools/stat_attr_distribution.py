import argparse
import pandas as pd
import os


def stat_celeba_attr(attr_path, partition_path=None, split=None):
    # 读取属性文件
    df = pd.read_csv(attr_path, sep='\s+', skiprows=1, index_col=0)
    # -1/1转为0/1
    df = (df == 1).astype(int)
    if partition_path and split:
        part_df = pd.read_csv(partition_path, sep='\s+', header=None, index_col=0)
        if split == 'train':
            img_names = part_df[part_df[1] == 0].index.tolist()
        elif split == 'val':
            img_names = part_df[part_df[1] == 1].index.tolist()
        elif split == 'test':
            img_names = part_df[part_df[1] == 2].index.tolist()
        else:
            raise ValueError('split must be train/val/test')
        # 只保留有属性的图片
        img_names = [n for n in img_names if n in df.index]
        df = df.loc[img_names]
    print(f"共{df.shape[0]}张图片，{df.shape[1]}个属性")
    results = []
    for col in df.columns:
        cnt_1 = df[col].sum()
        cnt_0 = len(df) - cnt_1
        ratio_1 = cnt_1 / len(df)
        ratio_0 = cnt_0 / len(df)
        balance = min(ratio_0, ratio_1)
        results.append({
            'attr': col,
            'cnt_0': cnt_0,
            'cnt_1': cnt_1,
            'ratio_0': ratio_0,
            'ratio_1': ratio_1,
            'balance': balance
        })
    # 按平衡度排序
    results = sorted(results, key=lambda x: -x['balance'])
    print(f"\n属性分布（按平衡度降序）：")
    print(f"{'属性':<18}{'0数':>8}{'1数':>8}{'0比例':>10}{'1比例':>10}{'平衡度':>10}")
    for r in results:
        print(f"{r['attr']:<18}{r['cnt_0']:>8}{r['cnt_1']:>8}{r['ratio_0']:>10.3f}{r['ratio_1']:>10.3f}{r['balance']:>10.3f}")
    print(f"\n最平衡的属性是: {results[0]['attr']}，平衡度={results[0]['balance']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='统计CelebA等数据集属性分布，辅助选择平衡目标属性')
    parser.add_argument('--attr_path', type=str, required=True, help='属性文件路径，如list_attr_celeba.txt')
    parser.add_argument('--partition_path', type=str, default=None, help='划分文件路径，如list_eval_partition.txt')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test'], help='只统计某个划分')
    args = parser.parse_args()
    stat_celeba_attr(args.attr_path, args.partition_path, args.split)

if __name__ == '__main__':
    main() 