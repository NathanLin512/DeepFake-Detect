import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        初始化 GRU 模型
        :param input_size: 每個輸入向量的特徵數量 (對應 HDDM 輸出的維度)
        :param hidden_size: GRU 隱藏層大小
        :param num_layers: GRU 的層數
        :param num_classes: 最終分類類別數量（例如 2：real 和 fake）
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU 層
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 全連接層 (將隱藏層輸出映射到分類)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向傳播
        :param x: shape = (batch_size, sequence_length, input_size)
        :return: shape = (batch_size, num_classes)
        """
        # 初始化隱藏層的初始狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU 前向計算
        out, _ = self.gru(x, h0)  # out shape: (batch_size, sequence_length, hidden_size)

        # 取最後一個時間步的隱藏狀態
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # 全連接層分類
        out = self.fc(out)  # shape: (batch_size, num_classes)
        return out
