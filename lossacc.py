import torch
import numpy as np
import matplotlib.pyplot as plt

# Load data from the checkpoint file
checkpoint = torch.load('checkpoint.pth')
losses = checkpoint['losses']
accuracies = checkpoint['accuracies']

# Plotting loss and accuracy
plt.figure(figsize=(10, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

 #Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(len(accuracies)), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()