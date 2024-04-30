import numpy
import torch
import os
import random
from collections import defaultdict
from tqdm import tqdm

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
    return numpy.asarray(t)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count + 1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield numpy.asarray(t)

# 超参数
hidden_dim = 50  # 隐藏层维度
learning_rate = 0.001  # 学习率

data_path = os.path.join('C:\\Users\\lenovo\\PycharmProjects\\pythonProject2', 'ml-1m.txt')
user_count, item_count, user_ratings = load_data(data_path)

# 创建模型
model = RecommendationModel(user_count, item_count, hidden_dim)

optimizer = torch.optim.SGD([model.user_emb_w, model.item_emb_w, model.fc1.weight, model.fc2.weight], lr=learning_rate, momentum=0.9, weight_decay=0.001)

# Check if there's a previous checkpoint
if os.path.exists('checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    losses = checkpoint.get('losses', [])  # Get losses if available, otherwise set to empty list
    accuracies = checkpoint.get('accuracies', [])  # Get accuracies if available, otherwise set to empty list
else:
    start_epoch = 0
    losses = []
    accuracies = []

# 生成测试集
user_ratings_test = generate_test(user_ratings)

# Training
for epoch_idx in range(start_epoch, 10):
    loss_mean = 0
    correct_predictions = 0
    total_predictions = 0
    # tqdm 进度条
    with tqdm(total=5000, desc=f'Epoch {epoch_idx}', unit='batch') as pbar:
        for batch_idx in range(5000):
            uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512)
            u, i, j = torch.tensor(uij[:, 0], dtype=torch.int64), torch.tensor(uij[:, 1], dtype=torch.int64), torch.tensor(
                uij[:, 2], dtype=torch.int64)
            optimizer.zero_grad()
            scores_i, scores_j = model(u, i, j)
            bprloss = -torch.mean(torch.log(torch.sigmoid(scores_i - scores_j)))
            bprloss.backward()
            optimizer.step()
            loss_mean += bprloss.data

            # Calculate accuracy
            correct_predictions += torch.sum(scores_i > scores_j).item()
            total_predictions += len(scores_i)
            pbar.update(1)
    # Calculate accuracy for this epoch
    accuracy = correct_predictions / total_predictions

    preds = []
    for uij_test in generate_test_batch(user_ratings, user_ratings_test, item_count):
        u, i, j = torch.tensor(uij_test[:, 0], dtype=torch.int64), torch.tensor(uij_test[:, 1],
                                                                                    dtype=torch.int64), torch.tensor(
            uij_test[:, 2], dtype=torch.int64)
        pred_i, pred_j = model(u, i, j)
        pred = torch.sigmoid(pred_i - pred_j)
        preds.extend(pred.tolist())
    preds_num = len(preds)
    print("Epoch {}, loss:{}, acc:{}".format(epoch_idx, loss_mean * 1.0 / 5000, accuracy))

    # Save losses and accuracies to checkpoint
    losses.append(loss_mean.item() / 5000)
    accuracies.append(accuracy)
    torch.save({
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'accuracies': accuracies
    }, 'checkpoint.pth')





