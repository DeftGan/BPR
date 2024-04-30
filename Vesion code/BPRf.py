import numpy as np
import torch
import os
import random
from collections import defaultdict
from tqdm import tqdm

from BPR2 import user_ratings_test


class RecommendationModel(torch.nn.Module):
    def __init__(self, user_count, item_count, hidden_dim):
        super(RecommendationModel, self).__init__()
        self.user_emb_w = torch.nn.Parameter(torch.rand((user_count + 1, hidden_dim), dtype=torch.float32))
        self.item_emb_w = torch.nn.Parameter(torch.rand((item_count + 1, hidden_dim), dtype=torch.float32))
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)  # 添加一个全连接层
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 再添加一个全连接层

    def forward(self, u, i, j):
        u_emb = self.user_emb_w[u]
        i_emb = self.item_emb_w[i]
        j_emb = self.item_emb_w[j]
        # 进行隐藏层的前向传播
        u_emb = self.fc1(u_emb)
        u_emb = torch.relu(u_emb)
        u_emb = self.fc2(u_emb)
        u_emb = torch.relu(u_emb)
        # 计算输出
        scores_i = torch.sum(u_emb * i_emb, dim=1)
        scores_j = torch.sum(u_emb * j_emb, dim=1)
        return scores_i, scores_j


def load_data(data_path):
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print("max_u_id:", max_u_id)
    print("max_i_id:", max_i_id)
    return max_u_id, max_i_id, user_ratings


def generate_test(user_ratings):
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(list(user_ratings[u]), 1)[0]
    return user_test


def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    t = []
    user_keys = list(user_ratings.keys())  # Convert dictionary view to list
    for b in range(batch_size):
        u = random.sample(user_keys, 1)[0]
        i = random.sample(list(user_ratings[u]), 1)[0]  # Convert set to list
        while i == user_ratings_test[u]:
            i = random.sample(list(user_ratings[u]), 1)[0]
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    return np.asarray(t)


def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count + 1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield np.asarray(t)


# 超参数
hidden_dim = 50  # 隐藏层维度
learning_rate = 0.001  # 学习率
num_clients = 5  # 模拟的客户端数量

data_path = os.path.join('C:\\Users\\lenovo\\PycharmProjects\\pythonProject2', 'ml-1m.txt')
user_count, item_count, user_ratings = load_data(data_path)

# 创建模型
global_model = RecommendationModel(user_count, item_count, hidden_dim)
optimizer = torch.optim.SGD(
    [global_model.user_emb_w, global_model.item_emb_w, global_model.fc1.weight, global_model.fc2.weight],
    lr=learning_rate, momentum=0.9, weight_decay=0.001)

# 模拟联邦学习
losses = []
accuracies = []
for epoch_idx in range(10):  # 假设每个客户端只训练一轮
    local_models = []
    for client_id in range(num_clients):
        # 创建客户端模型
        local_model = RecommendationModel(user_count, item_count, hidden_dim)
        local_model.load_state_dict(global_model.state_dict())  # 每个客户端初始模型均与全局模型相同
        optimizer_client = torch.optim.SGD(
            [local_model.user_emb_w, local_model.item_emb_w, local_model.fc1.weight, local_model.fc2.weight],
            lr=learning_rate, momentum=0.9, weight_decay=0.001)

        # 客户端训练
        for batch_idx in range(5000):
            uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512)
            u, i, j = torch.tensor(uij[:, 0], dtype=torch.int64), torch.tensor(uij[:, 1],
                                                                               dtype=torch.int64), torch.tensor(
                uij[:, 2], dtype=torch.int64)
            optimizer_client.zero_grad()
            scores_i, scores_j = local_model(u, i, j)
            bprloss = -torch.mean(torch.log(torch.sigmoid(scores_i - scores_j)))
            bprloss.backward()
            optimizer_client.step()

        local_models.append(local_model)

    # 聚合更新
    with torch.no_grad():
        for param_global, param_local in zip(global_model.parameters(),
                                             zip(*[local_model.parameters() for local_model in local_models])):
            param_global.copy_(sum(local_param.data for local_param in param_local) / num_clients)

    # 在全局模型上评估
    loss_mean = 0
    correct_predictions = 0
    total_predictions = 0
    for batch_idx in range(5000):
        uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512)
        u, i, j = torch.tensor(uij[:, 0], dtype=torch.int64), torch.tensor(uij[:, 1], dtype=torch.int64), torch.tensor(
            uij[:, 2], dtype=torch.int64)
        scores_i, scores_j = global_model(u, i, j)
        loss = -torch.mean(torch.log(torch.sigmoid(scores_i - scores_j)))
        loss_mean += loss.item()

        # Calculate accuracy
        correct_predictions += torch.sum(scores_i > scores_j).item()
        total_predictions += len(scores_i)

    accuracy = correct_predictions / total_predictions

    # Print loss and accuracy for this epoch
    print(f"Epoch {epoch_idx + 1}: Loss: {loss_mean / 5000}, Accuracy: {accuracy}")

    # Save loss and accuracy for plotting
    losses.append(loss_mean / 5000)
    accuracies.append(accuracy)

    # 保存 loss 和 accuracy 到文件
    np.save('losses.npy', losses)
    np.save('accuracies.npy', accuracies)