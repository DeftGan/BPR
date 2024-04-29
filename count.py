import torch

# Define the model architecture
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

# Load the checkpoint
checkpoint_path = 'checkpoint.pth'
checkpoint = torch.load(checkpoint_path)

# Extract necessary information
final_loss = checkpoint.get('losses', [])[-1] if 'losses' in checkpoint else None
final_accuracy = checkpoint.get('accuracies', [])[-1] if 'accuracies' in checkpoint else None

# Display final training loss and accuracy
print("Final training loss:", final_loss)
print("Final training accuracy:", final_accuracy)