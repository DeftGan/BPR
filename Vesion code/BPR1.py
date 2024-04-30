import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyexpat import model


class BPR(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=32):
        super(BPR, self).__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        user_embedding = self.user_emb(user_ids)
        pos_item_embedding = self.item_emb(pos_item_ids)
        neg_item_embedding = self.item_emb(neg_item_ids)

        pos_scores = torch.sum(user_embedding * pos_item_embedding, dim=1)
        neg_scores = torch.sum(user_embedding * neg_item_embedding, dim=1)

        return pos_scores, neg_scores


def train_bpr_custom(data, num_users, num_items, epochs=30, learning_rate=0.05):
    model = BPR(num_users, num_items)
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        np.random.shuffle(data)
        for user_id, pos_item_id in data:  # 每行只包含用户ID和正样本物品ID
            # 随机选择一个负样本（未被用户评分的物品）
            neg_item_id = np.random.randint(num_items)
            while (user_id, neg_item_id) in data:  # 确保负样本不是用户已经评分过的物品
                neg_item_id = np.random.randint(num_items)

            user_id_tensor = torch.tensor([user_id], dtype=torch.long)
            pos_item_id_tensor = torch.tensor([pos_item_id], dtype=torch.long)
            neg_item_id_tensor = torch.tensor([neg_item_id], dtype=torch.long)

            optimizer.zero_grad()
            pos_scores, neg_scores = model(user_id_tensor, pos_item_id_tensor, neg_item_id_tensor)
            loss = criterion(pos_scores, neg_scores, torch.ones(1))
            loss.backward()
            optimizer.step()


def recommend(model, dataset, user_ids, k=3):
    for user_id in user_ids:
        # 获取用户已经交互过的物品
        known_positives = set(dataset[dataset[:, 1] == user_id][:, 0])

        # 对所有未交互过的物品进行预测得分
        scores = model.user_emb(torch.tensor([user_id])).dot(model.item_emb.weight.t())
        scores = scores.squeeze().detach().numpy()

        # 对得分进行排序，选择前k个物品进行推荐
        top_items = np.argsort(-scores)
        recommended_items = [item for item in top_items if item not in known_positives][:k]

        print("User %s" % user_id)
        print(" Known positives:", known_positives)
        print(" Recommended:", recommended_items)


# 加载数据集
file_path = 'C:\\Users\\lenovo\\PycharmProjects\\pythonProject2\\u.txt'  # 替换为你的数据集文件路径
dataset = np.loadtxt(file_path, dtype=int, usecols=(0, 1))

# 调整索引以从零开始
dataset[:, 0] -= 1  # 将ItemID减去1
dataset[:, 1] -= 1  # 将UserID减去1

# 获取用户数和物品数
num_users = np.max(dataset[:, 1]) + 1
num_items = np.max(dataset[:, 0]) + 1

# 训练模型
train_bpr_custom(dataset, num_users, num_items)

# 选择用户并进行推荐
user_ids = [0, 1, 2]  # 选择一些用户进行推荐
recommend(model, dataset, user_ids)