import torch
import random
from collections import defaultdict


class RecommendationModel(torch.nn.Module):
    def __init__(self, user_count, item_count, hidden_dim):
        super(RecommendationModel, self).__init__()
        self.user_emb_w = torch.nn.Parameter(torch.rand((user_count + 1, hidden_dim), dtype=torch.float32))
        self.item_emb_w = torch.nn.Parameter(torch.rand((item_count + 1, hidden_dim), dtype=torch.float32))
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, u, i, j):
        u_emb = self.user_emb_w[u]
        i_emb = self.item_emb_w[i]
        j_emb = self.item_emb_w[j]

        u_emb = self.fc1(u_emb)
        u_emb = torch.relu(u_emb)
        u_emb = self.fc2(u_emb)
        u_emb = torch.relu(u_emb)

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


def predict_top_items(model, user_id, item_count, top_k=5):
    user_tensor = torch.tensor([user_id], dtype=torch.int64)
    item_scores = []
    for item_id in range(1, item_count + 1):
        if item_id not in user_ratings[user_id]:
            item_tensor = torch.tensor([item_id], dtype=torch.int64)
            score_i, _ = model(user_tensor, item_tensor, item_tensor)
            item_scores.append((item_id, score_i.item()))
    sorted_item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
    top_items = [item_id for item_id, _ in sorted_item_scores[:top_k]]
    return top_items


# Load the dataset and dynamically calculate the number of users and items
data_path = 'ml-1m.txt'  # Assume the dataset path is ml-1m.txt
user_count, item_count, user_ratings = load_data(data_path)
print("Number of users:", user_count)
print("Number of items:", item_count)

# Load the trained model
hidden_dim = 50  # Assume the hidden layer dimension is 50
model = RecommendationModel(user_count, item_count, hidden_dim)
checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Choose 5 random user IDs
random_user_ids = random.sample(range(1, user_count + 1), 5)  # Randomly select 5 user IDs

# Predict the top items each user might like
for user_id in random_user_ids:
    top_items = predict_top_items(model, user_id, item_count)
    random_top_items = random.sample(top_items, min(5, len(top_items)))  # Randomly select 5 items
    print(f"User {user_id} might like items: {random_top_items}")



