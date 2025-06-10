import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims=None):
        super(AutoEncoder, self).__init__()
        if hidden_dims is None:
            # Default architecture similar to DEC paper for MNIST-like data
            # Input -> 500 -> 500 -> 2000 -> embedding_dim
            hidden_dims = [500, 500, 2000]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, embedding_dim))
        # No ReLU on the embedding layer output as per DEC paper
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = embedding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        decoder_layers.append(nn.Tanh()) # Assuming input is normalized to [-1, 1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Flatten input if it's not already
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, embedding_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        # Initialize cluster centers as a learnable parameter
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        # Initialize centers with Xavier Glorot initialization
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z):
        """
        Compute the soft assignment q_ij.
        z: embedding of data points, shape (batch_size, embedding_dim)
        """
        # ||z_i - mu_j||^2
        # (batch_size, 1, embedding_dim) - (1, n_clusters, embedding_dim) -> (batch_size, n_clusters, embedding_dim)
        diff_sq = torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers.unsqueeze(0), 2), dim=2)
        
        # q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
        numerator = (1.0 + diff_sq / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        
        # q_ij = q_ij / sum_k(q_ik)  (normalize over clusters for each sample)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q

class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, alpha=1.0):
        super(DEC, self).__init__()
        self.encoder = autoencoder.encoder
        self.n_clusters = n_clusters
        self.embedding_dim = autoencoder.embedding_dim
        self.clustering_layer = ClusteringLayer(n_clusters, self.embedding_dim, alpha)

    def forward(self, x):
        # Flatten input if necessary
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)
        q = self.clustering_layer(z)
        return q

    @staticmethod
    def target_distribution(q):
        """
        Compute the target distribution p_ij.
        q: soft assignments from the clustering layer, shape (batch_size, n_clusters)
        """
        # p_ij = (q_ij^2 / f_j) / sum_k(q_ik^2 / f_k)
        # f_j = sum_i q_ij (sum of soft assignments for cluster j, over all samples)
        # For batch-wise update, f_j is calculated on the current batch.
        # The original paper calculates f_j over the entire dataset, but for SGD, batch-wise is common.
        # Let's stick to the paper's intention of using full dataset properties for P.
        # However, for practical mini-batch training, P is derived from Q of the current batch.
        
        weight = q**2 / torch.sum(q, dim=0) # q_ij^2 / f_j (where f_j is sum_i q_ij for the batch)
        p = (weight.t() / torch.sum(weight, dim=1)).t() # Normalize over clusters
        return p.detach() # Detach as P is fixed during KL divergence minimization for a given Q

def pretrain_autoencoder(ae_model, data_loader, epochs, learning_rate, device='cuda'):
    print(f"开始预训练 AutoEncoder (DEC 初始化)...")
    ae_model.to(device)
    optimizer = Adam(ae_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        batch_iterator = tqdm(data_loader, desc=f"AE Pretrain Epoch {epoch+1}/{epochs}", leave=False)
        for data_batch in batch_iterator:
            if isinstance(data_batch, (list, tuple)):
                inputs = data_batch[0].to(device)
            else:
                inputs = data_batch.to(device) # Assuming data_batch is just x

            if inputs.dim() > 2:
                inputs_flattened = inputs.view(inputs.size(0), -1)
            else:
                inputs_flattened = inputs
            
            optimizer.zero_grad()
            reconstructions = ae_model(inputs_flattened)
            loss = criterion(reconstructions, inputs_flattened)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"AE 预训练 Epoch [{epoch+1}/{epochs}], 平均损失: {avg_epoch_loss:.4f}")
    print("AutoEncoder 预训练完成。")
    return ae_model

def train_dec(dec_model, autoencoder_model, data_loader, epochs, learning_rate, n_clusters, device='cuda', 
              update_interval=100, tol=1e-3, ae_pretrain_epochs=50, ae_lr=1e-3):
    
    # 1. Pretrain AutoEncoder if not already done (or load weights)
    # For simplicity, we assume pretraining happens here or autoencoder_model is already pretrained.
    # If `autoencoder_model` is fresh, it should be trained.
    # Let's add a simple check: if its parameters have requires_grad=False, it might be an issue.
    # A better approach would be to pass pretrained AE weights or a flag.
    # For now, let's assume ae_model needs pretraining if not specifically handled outside.
    
    # --- Pretrain AE (if not already trained) ---
    # This is a placeholder. In a real pipeline, you'd train it once.
    # We can train it here for demonstration.
    print("检查是否需要预训练AE...")
    # A simple heuristic: if AE has not been moved to device and trained.
    # This is not robust. User should manage pretraining ideally.
    # For this example, we'll pretrain it.
    pretrain_autoencoder(autoencoder_model, data_loader, epochs=ae_pretrain_epochs, learning_rate=ae_lr, device=device)
    
    # --- Initialize DEC model ---
    dec_model.to(device)
    
    # --- Initialize cluster centers using K-means on AE features ---
    print("使用 K-means 初始化聚类中心...")
    all_features_list = []
    autoencoder_model.eval() # Set AE to eval mode for feature extraction
    with torch.no_grad():
        for data_batch in data_loader:
            if isinstance(data_batch, (list, tuple)):
                inputs = data_batch[0].to(device)
            else:
                inputs = data_batch.to(device)
            
            if inputs.dim() > 2:
                inputs_flattened = inputs.view(inputs.size(0), -1)
            else:
                inputs_flattened = inputs
            features = autoencoder_model.encode(inputs_flattened)
            all_features_list.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features_list, axis=0)
    print(f"提取的特征形状: {all_features.shape}")
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0) # n_init=20 as in DEC paper
    predicted_labels_kmeans = kmeans.fit_predict(all_features)
    cluster_centers_kmeans = kmeans.cluster_centers_
    
    # Assign K-means centroids to the DEC clustering layer
    dec_model.clustering_layer.cluster_centers.data = torch.from_numpy(cluster_centers_kmeans).to(device)
    print("聚类中心初始化完成。")

    # --- DEC Training (Fine-tuning) ---
    optimizer = Adam(dec_model.parameters(), lr=learning_rate)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean') # KL(P || Q)

    last_predicted_labels = None
    print(f"开始 DEC 训练，共 {epochs} 个 epochs...")

    for epoch in range(epochs):
        dec_model.train() # Set DEC model (encoder + clustering layer) to train mode
        total_loss = 0
        num_batches = len(data_loader)
        
        batch_iterator = tqdm(data_loader, desc=f"DEC Epoch {epoch+1}/{epochs}", leave=False)

        for i, data_batch in enumerate(batch_iterator):
            if isinstance(data_batch, (list, tuple)):
                inputs = data_batch[0].to(device)
            else:
                inputs = data_batch.to(device)

            if inputs.dim() > 2:
                inputs_flattened = inputs.view(inputs.size(0), -1)
            else:
                inputs_flattened = inputs

            # Forward pass: get soft assignments q
            q = dec_model(inputs_flattened)

            # Update target distribution p (only periodically, or if first iteration)
            # The original paper computes P over the whole dataset.
            # For mini-batch, P is often updated based on Q from current batch,
            # or less frequently using Q from all data.
            # Here we use batch-wise P for simplicity.
            with torch.no_grad(): # p should be detached
                p = DEC.target_distribution(q)
            
            # Compute KL divergence loss: KL(P || Q)
            # KLDivLoss expects log probabilities for Q.
            loss = kl_loss_fn(q.log(), p) # target is p, input is q.log()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            batch_iterator.set_postfix({"KL_loss": f"{loss.item():.4f}"})

        avg_epoch_loss = total_loss / num_batches
        print(f"DEC Epoch [{epoch+1}/{epochs}], 平均 KL 损失: {avg_epoch_loss:.4f}")

        # --- Check for convergence (optional, based on label changes) ---
        # This requires getting predictions for the whole dataset, can be slow.
        # For a full implementation, one would do this less frequently.
        if (epoch + 1) % update_interval == 0 or epoch == epochs - 1: # Check less frequently
            print("检查收敛性 (计算当前聚类分配)...")
            current_features_list = []
            current_labels_list = []
            dec_model.eval()
            with torch.no_grad():
                for data_batch_eval in data_loader: # Use full loader or a test/validation loader
                    if isinstance(data_batch_eval, (list, tuple)):
                        inputs_eval = data_batch_eval[0].to(device)
                    else:
                        inputs_eval = data_batch_eval.to(device)
                    
                    if inputs_eval.dim() > 2:
                        inputs_eval_flat = inputs_eval.view(inputs_eval.size(0), -1)
                    else:
                        inputs_eval_flat = inputs_eval
                    
                    q_eval = dec_model(inputs_eval_flat)
                    preds_eval = torch.argmax(q_eval, dim=1)
                    current_labels_list.append(preds_eval.cpu().numpy())
            
            current_predicted_labels = np.concatenate(current_labels_list)

            if last_predicted_labels is not None:
                delta_label = np.sum(current_predicted_labels != last_predicted_labels).astype(np.float32) / current_predicted_labels.shape[0]
                print(f"Epoch {epoch+1}: 标签变化百分比: {delta_label*100:.3f}%")
                if delta_label < tol:
                    print(f"达到收敛阈值 ({tol*100:.3f}%)。停止训练。")
                    break
            last_predicted_labels = current_predicted_labels
            
    print("DEC 训练完成。")
    return dec_model


if __name__ == '__main__':
    # --- 示例参数 (以 MNIST 为例) ---
    INPUT_DIM = 784          # MNIST: 28*28
    EMBEDDING_DIM = 10       # DEC paper uses 10 for MNIST
    N_CLUSTERS = 10          # MNIST has 10 classes/clusters
    AE_HIDDEN_DIMS = [500, 500, 2000] # Encoder: 784 -> 500 -> 500 -> 2000 -> 10
    
    NUM_SAMPLES = 1000 # Reduced for quick demo
    BATCH_SIZE = 256   # DEC paper uses 256
    
    AE_PRETRAIN_EPOCHS = 10 # Reduced for demo; paper might use 50-100k iterations (approx 200-400 epochs for 60k MNIST)
    AE_LR = 1e-3
    
    DEC_EPOCHS = 20 # Reduced for demo; paper might use 20k iterations (approx 80 epochs for 60k MNIST)
    DEC_LR = 1e-3 # Learning rate for DEC stage; paper might use 0.01, then 0.001
    UPDATE_INTERVAL = 1 # Check convergence every epoch for demo. Paper: 140 iterations (less than 1 epoch).
    CONVERGENCE_TOL = 0.001 # 0.1% label change

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    # --- 创建 AutoEncoder 模型 ---
    autoencoder = AutoEncoder(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_dims=AE_HIDDEN_DIMS)
    print(f"AutoEncoder 模型已创建: Input={INPUT_DIM}, Embedding={EMBEDDING_DIM}, Hidden={AE_HIDDEN_DIMS}")
    
    # --- 创建示例数据和 DataLoader ---
    print(f"创建示例数据: {NUM_SAMPLES} 个样本, 特征维度={INPUT_DIM}")
    # Simulating normalized MNIST data [-1, 1]
    dummy_data_tensor = torch.rand(NUM_SAMPLES, INPUT_DIM) * 2 - 1 
    
    from torch.utils.data import TensorDataset, DataLoader
    dummy_dataset = TensorDataset(dummy_data_tensor)
    # For DEC, data_loader is used for AE pretraining, K-means init, and DEC training
    # Shuffle should be True for training phases.
    dummy_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False) # drop_last=False for eval parts
    print(f"DataLoader 已创建: Batch Size={BATCH_SIZE}")

    # --- 创建 DEC 模型 ---
    # The autoencoder passed here is just for structure; its weights will be trained in pretrain_autoencoder
    dec_model_instance = DEC(autoencoder=AutoEncoder(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_dims=AE_HIDDEN_DIMS), 
                             n_clusters=N_CLUSTERS)
    print(f"DEC 模型已创建: Clusters={N_CLUSTERS}, Embedding Dim={EMBEDDING_DIM}")
    
    # --- 训练 DEC ---
    # The `autoencoder` instance defined above will be pretrained inside `train_dec`
    trained_dec_model = train_dec(
        dec_model=dec_model_instance,
        autoencoder_model=autoencoder, # This AE instance will be trained
        data_loader=dummy_loader,
        epochs=DEC_EPOCHS,
        learning_rate=DEC_LR,
        n_clusters=N_CLUSTERS,
        device=DEVICE,
        update_interval=UPDATE_INTERVAL,
        tol=CONVERGENCE_TOL,
        ae_pretrain_epochs=AE_PRETRAIN_EPOCHS,
        ae_lr=AE_LR
    )

    # --- 获取最终聚类结果 ---
    print("\n获取最终聚类结果...")
    trained_dec_model.eval()
    final_features_list = []
    final_labels_list = []
    with torch.no_grad():
        for data_batch in dummy_loader: # Use the same loader (or a test loader)
            inputs = data_batch[0].to(DEVICE)
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)
            
            q_final = trained_dec_model(inputs)
            preds_final = torch.argmax(q_final, dim=1)
            
            # Optionally, get features from the encoder part
            # z_final = trained_dec_model.encoder(inputs) 
            # final_features_list.append(z_final.cpu().numpy())
            final_labels_list.append(preds_final.cpu().numpy())

    final_predicted_labels = np.concatenate(final_labels_list)
    # final_extracted_features = np.concatenate(final_features_list) if final_features_list else None

    print(f"\n--- 最终聚类结果 ---")
    print(f"样本数量: {final_predicted_labels.shape[0]}")
    print(f"聚类标签数量: {len(np.unique(final_predicted_labels))}")
    print(f"聚类标签 (前20个): {final_predicted_labels[:20]}")
    # if final_extracted_features is not None:
    #     print(f"提取的特征形状: {final_extracted_features.shape}")

    print("\nDEC 示例运行完成。")
    print("注意: 示例使用随机数据和少量epoch，实际效果需在真实数据集上调整参数并充分训练。") 