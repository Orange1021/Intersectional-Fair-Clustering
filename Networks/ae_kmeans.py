import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, representation_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = representation_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, representation_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh() # 更改为 Tanh，以适应[-1, 1]范围的数据，如常用方式归一化的MNIST
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        # 确保输入是二维的 [batch_size, features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)

def train_autoencoder(ae_model, data_loader, epochs, learning_rate, device='cuda'):
    """
    训练AutoEncoder模型。
    """
    ae_model.to(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"开始训练 AutoEncoder，共 {epochs} 个 epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        for data_batch in data_loader:
            if isinstance(data_batch, (list, tuple)):
                inputs = data_batch[0].to(device)
            else:
                inputs = data_batch.to(device)

            if inputs.dim() > 2: # 例如图像数据
                inputs_flattened = inputs.view(inputs.size(0), -1)
            else:
                inputs_flattened = inputs

            optimizer.zero_grad()
            reconstructions = ae_model(inputs_flattened)
            loss = criterion(reconstructions, inputs_flattened)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"AE 训练 Epoch [{epoch+1}/{epochs}], 平均损失: {avg_epoch_loss:.4f}")
    print("AutoEncoder 训练完成。")
    return ae_model

def ae_kmeans_cluster(data_loader, ae_model, n_clusters, device='cuda'):
    """
    使用AutoEncoder提取特征后进行k-means聚类。

    参数:
        data_loader: 输入数据的DataLoader。
        ae_model: 预训练的AutoEncoder模型。
        n_clusters: k-means的簇数量。
        device: 模型运行的设备 ('cuda' 或 'cpu')。

    返回:
        cluster_labels: 每个数据点的聚类标签。
        cluster_centers: 聚类中心的坐标。
        all_features_np: AutoEncoder提取的所有特征。
    """
    ae_model.eval()
    ae_model.to(device)
    all_features = []
    print("开始使用 AE 提取特征...")
    with torch.no_grad():
        for data_batch in data_loader:
            if isinstance(data_batch, (list, tuple)):
                inputs = data_batch[0].to(device)
            else:
                inputs = data_batch.to(device)

            if inputs.dim() > 2: # 例如图像数据
                inputs_flattened = inputs.view(inputs.size(0), -1)
            else:
                inputs_flattened = inputs
            
            features_batch = ae_model.encode(inputs_flattened)
            all_features.append(features_batch.cpu().numpy())
    
    all_features_np = np.concatenate(all_features, axis=0)
    print(f"特征提取完成，特征形状: {all_features_np.shape}")

    print(f"开始使用 k-means 聚类，k={n_clusters}...")
    # 在 scikit-learn 0.23+ 版本中，KMeans 的 n_init 默认为 10。
    # 如果您的版本较旧，可能需要明确设置 n_init，例如 n_init=10。
    # 对于新版本，可以使用 n_init='auto'，但为了兼容性，这里保留数字。
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10) 
    cluster_labels = kmeans.fit_predict(all_features_np)
    cluster_centers = kmeans.cluster_centers_
    print("k-means 聚类完成。")

    return cluster_labels, cluster_centers, all_features_np

if __name__ == '__main__':
    # --- 示例参数 ---
    INPUT_DIM = 784  # 例如: MNIST 图像 28*28
    REPRESENTATION_DIM = 64 
    N_CLUSTERS = 10
    NUM_SAMPLES = 500
    BATCH_SIZE = 64
    AE_EPOCHS = 10 # AE训练的轮数
    AE_LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用设备: {DEVICE}")

    # --- 创建示例AutoEncoder模型 ---
    autoencoder = AutoEncoder(input_dim=INPUT_DIM, representation_dim=REPRESENTATION_DIM)
    print(f"AutoEncoder模型已创建: 输入维度={INPUT_DIM}, 隐层维度={REPRESENTATION_DIM}")

    # --- 创建示例数据和DataLoader ---
    # 在实际应用中, 你需要加载你的真实数据集
    print(f"创建示例数据: {NUM_SAMPLES} 个样本, 特征维度={INPUT_DIM}")
    dummy_data_tensor = torch.rand(NUM_SAMPLES, INPUT_DIM) 
    # 如果是图像数据，可能是 [NUM_SAMPLES, 1, 28, 28]
    # dummy_data_tensor = torch.rand(NUM_SAMPLES, 1, 28, 28) 
    
    from torch.utils.data import TensorDataset, DataLoader
    dummy_dataset = TensorDataset(dummy_data_tensor)
    dummy_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"DataLoader已创建: 批大小={BATCH_SIZE}")

    # --- 训练AutoEncoder ---
    # 注意：在实际使用中，AE的输入维度需要和数据匹配。
    # 如果你的数据是图像 (例如 [N, C, H, W]), AE的输入层也应设计为接收扁平化的图像 (N, C*H*W)
    # 或者使用卷积AE。此示例使用全连接AE，输入应为二维。
    autoencoder = train_autoencoder(autoencoder, dummy_loader, epochs=AE_EPOCHS, learning_rate=AE_LEARNING_RATE, device=DEVICE)

    # --- 执行 AE + k-means 聚类 ---
    print(f"\n开始执行 AE + k-means 聚类 (k={N_CLUSTERS})...")
    # 假设AE已经训练完毕
    cluster_labels, cluster_centers, extracted_features = ae_kmeans_cluster(
        data_loader=dummy_loader, # 使用相同的loader提取特征，或者用一个不shuffle的测试loader
        ae_model=autoencoder,
        n_clusters=N_CLUSTERS,
        device=DEVICE
    )

    print(f"\n--- 聚类结果 ---")
    print(f"提取特征数量: {extracted_features.shape[0]}")
    print(f"每个特征的维度: {extracted_features.shape[1]}")
    print(f"聚类数量: {len(np.unique(cluster_labels))}")
    print(f"聚类标签 (前20个): {cluster_labels[:20]}")
    print(f"聚类中心形状: {cluster_centers.shape}")

    # --- 后续步骤 ---
    # 1. 将此模块集成到你的主训练/评估流程中。
    # 2. 根据你的具体数据集调整 input_dim 和 representation_dim。
    # 3. 如果是图像等高维数据，考虑使用卷积自编码器 (Convolutional AutoEncoder) 以获得更好的特征。
    # 4. 实现完整的评估流程，例如使用 NMI, ACC 等指标评估聚类效果。
    print("\n示例运行完成。")
