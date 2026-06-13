import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义线性层 - 用于学习指标权重
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        # 只保留权重参数，每个输入特征对应一个可学习的权重
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # 不使用bias，因为我们只需要学习权重
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化为均匀分布，确保所有权重起点相同
        nn.init.constant_(self.weight, 1.0 / self.weight.size(0))

    def forward(self, input):
        # 对每一列分别进行softmax归一化，确保权重和为1且都为正
        weight_normalized = torch.softmax(self.weight, dim=0)
        # 加权求和，不添加偏置
        return torch.matmul(input, weight_normalized)

# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层：CustomLinear层用于学习16个指标的权重
        self.fc1 = CustomLinear(16, 1)  # 输出维度改为1，直接得到加权后的综合得分
        
        # 后续层用于分类
        self.fc2 = nn.Linear(1, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 4)  # 输出4个类别

    def forward(self, x):
        # 第一层：通过CustomLinear得到加权综合得分
        x = self.fc1(x)  # (batch_size, 1)
        
        # 后续层进行非线性变换和分类
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x
    
    def get_weights(self):
        """获取归一化后的指标权重"""
        weights = self.fc1.weight.data  # (16, 1)
        # 对权重进行softmax归一化，确保和为1
        norm_weights = F.softmax(weights.squeeze(), dim=0)  # (16,)
        return norm_weights.detach().cpu().numpy()


class Net_GRU(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, num_classes=4,
                 dropout=0.2, bidirectional=True):
        super(Net_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Keep the same interpretable weight interface as Net: (16, 1).
        # The raw parameters are normalized with softmax before every use.
        self.fc1 = CustomLinear(input_size, 1)

        # Treat the 16 indicators as a short sequence. For each indicator, the
        # GRU sees its original value and its softmax-normalized weight.
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_size + 1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _normalized_weights(self):
        return F.softmax(self.fc1.weight.squeeze(-1), dim=0)

    def forward(self, x):
        # x: (batch_size, 16), output: (batch_size, 4)
        weights = self._normalized_weights()  # (16,), sum == 1
        weighted_score = torch.matmul(x, weights.unsqueeze(1))  # (batch_size, 1)

        batch_size = x.size(0)
        value_seq = x.unsqueeze(-1)  # (batch_size, 16, 1)
        weight_seq = weights.view(1, self.input_size, 1).expand(batch_size, -1, -1)
        gru_input = torch.cat([value_seq, weight_seq], dim=-1)  # (batch_size, 16, 2)

        gru_output, _ = self.gru(gru_input)
        seq_feature = gru_output[:, -1, :]
        feature = torch.cat([seq_feature, weighted_score], dim=1)
        return self.classifier(feature)

    def get_weights(self):
        """Get softmax-normalized indicator weights, whose sum is 1."""
        norm_weights = self._normalized_weights()
        return norm_weights.detach().cpu().numpy()


# class MLP(nn.Module):

#     def __init__(self):
#         super(MLP, self).__init__()
#         # self.fc1 = CustomLinear(16, 32)
#         self.fc1 = nn.Linear(16, 64)
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(64, 64)
#         self.dropout2 = nn.Dropout(0.1)
#         self.fc3 = nn.Linear(64, 16) # fc3输出的维度为16，将其归一化后表示16个指标体系的权重
#         self.dropout3 = nn.Dropout(0.1)
#         self.fc4 = nn.Linear(16, 4)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc3(x))
#         x = self.dropout3(x)
#         x = self.fc4(x)
#         return x

#     def get_weights(self):
#         weights = self.fc3.weight.data
#         mean_weights = torch.mean(weights, dim=1) # (16,)
#         norm_weights = F.softmax(mean_weights, dim=0) # (16,)
#         return norm_weights.detach().cpu().numpy()
