import os
import pickle


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# # 读取.pkl文件
# with open('/home/myli/RL_Optimizer/RobustLight/inferences/data_jn_1.pkl', 'rb') as f:  # 注意模式是'rb'（二进制读取）
#     data = pickle.load(f)

# states = data[0]['lane_queue_in_part']
#state_s = data[2]['lane_queue_in_part']
def generate_data(states):
    seq_len = 5
    mask = [0,1,2]
    X_seq, X_curr, y = [], [], []
    for i in range(seq_len, len(states)):
        seq = states[i-seq_len: i]
        current = np.delete(states[i], mask)
        target = np.array(states[i])[mask]

        X_seq.append(seq)
        X_curr.append(current)
        y.append(target)
    return np.array(X_seq), np.array(X_curr), np.array(y)

# X_seq, X_curr, y  = generate_data(states)


# X_seq.shape
# (863995, 5, 12)
# X_curr.shape
# (863995, 9)
# y.shape
# (863995, 3)

# X_seq.shape
# (10000, 3, 24)
# X_curr.shape
# (10000, 21)
# y.shape
# (10000, 3)


# 数据集划分
# X_train, X_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
#     X_seq, X_curr, y, test_size=0.2, random_state=42
# )

# 自定义位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        return x + self.pe[:x.size(1)]

# Transformer模型定义
class TransformerRecovery(nn.Module):
    def __init__(self, input_dim=12, d_model=48, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + 9, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            #补全那三个空
            nn.Linear(32, 3)
        )

    def forward(self, seq, curr):
        # 序列嵌入与位置编码
        seq_emb = self.embedding(seq)  # (B,3,d_model)
        seq_emb = self.pos_encoder(seq_emb)
        
        # Transformer处理
        trans_out = self.transformer(seq_emb)  # (B,3,d_model)
        
        # 序列特征聚合（使用注意力池化）
        weights = torch.softmax(trans_out.mean(dim=2), dim=1).unsqueeze(2)
        seq_feat = torch.sum(trans_out * weights, dim=1)  # (B,d_model)
        
        # 特征融合
        fused = torch.cat([seq_feat, curr], dim=1)
        return self.feature_fusion(fused)

# 数据集类（保持相同）
class RecoveryDataset(Dataset):
    def __init__(self, sequences, currents, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.currents = torch.FloatTensor(currents)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.currents[idx], self.targets[idx]

# train_dataset = RecoveryDataset(X_train, Xc_train, y_train)
# test_dataset = RecoveryDataset(X_test, Xc_test, y_test)

# BATCH_SIZE = 64
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRecovery().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# 改进后的训练循环（增加梯度裁剪）
def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for seq, curr, y in loader:
        seq, curr, y = seq.to(device), curr.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, curr)
        loss = criterion(outputs, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * seq.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, curr, y in loader:
            seq, curr, y = seq.to(device), curr.to(device), y.to(device)
            outputs = model(seq, curr)
            total_loss += criterion(outputs, y).item() * seq.size(0)
    return total_loss / len(loader.dataset)

# # 训练过程（增加学习率调度）
# best_val_loss = float("inf")
# train_losses, val_losses = [], []

# #训练
# for epoch in range(0, 15):
#     train_loss = train_epoch(model, train_loader)
#     val_loss = evaluate(model, test_loader)
#     scheduler.step()
    
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
    
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#     torch.save(model.state_dict(), "%s.pt" % str(epoch))
    
#     print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.5f}")
    
#     # 动态早停策略
#     if epoch > 30 and val_loss > np.mean(val_losses[-10:]):
#         print("Early stopping triggered")
#         break

#测试
model.load_state_dict(torch.load('/home/myli/RL_Optimizer/RobustLight/inferences/49.pt',map_location=torch.device('cuda:0')))
model.eval()
model.to('cuda:0')
#随便取了一个
seq1 = test_dataset.sequences[4].reshape(1,5,12).to('cuda:0')
curr1 = test_dataset.currents[4].reshape(1,9).to('cuda:0')
target1 = np.int32(test_dataset.targets[4].numpy())

def pred_state(seq_states, curr_masked_state, model):
    #seq_states = [1,5,12]
    #curr_masked = [1,9]   前三个维度被删掉了

    #预测的前三个维度的数据
    pred_normalized = model(seq_states,curr_masked_state).cpu().detach().numpy()
    pred_normalized = np.int32(abs(pred_normalized.reshape(-1)))
    #拼接成完整的state
    pred_state = np.float32(np.concatenate((pred_normalized, np.int32(curr_masked_state.cpu().reshape(-1).numpy()))))
    return pred_state


pred_state = pred_state(seq1, curr1, model)
print(pred_state)
