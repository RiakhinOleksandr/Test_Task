# Importing necessary libraries and classes from model_training.py file
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_training import SiameseArrayDataset, EmbeddingNet, ContrastiveLoss, SiameseNet


if __name__ == "__main__":
    # Loading test data
    test_data = np.load("pairs_test_data.npz", allow_pickle=True)

    test_pairs = test_data["pairs"]
    test_labels = test_data["labels"]

    test_dataset = SiameseArrayDataset(test_pairs, test_labels)

    # We will be giving pairs of images by batches for testing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # If possible use GPU. It makes training much faster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting our model and loss function
    model = SiameseNet(EmbeddingNet()).to(device)
    criterion = ContrastiveLoss(margin=1.0)

    # Load the saved weights
    model.load_state_dict(torch.load("best_siamese_model.pth"))

    model.eval() # Transfer our model into evaluating mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            out1, out2 = model(img1, img2) # Getting embedding vectors of images from CNN

            euclidean_distance = F.pairwise_distance(out1, out2) # Calculating euclidean distance between embeddings
            predictions = (euclidean_distance < 0.5).float()  # Threshold to define 'same'. We setted it as 0.5
            correct += (predictions == labels).sum().item() # Calculating number of correct predictions
            total += labels.size(0) # Calculating number of total predictions

            loss = criterion(out1, out2, labels) # Calculating loss function
            test_loss += loss.item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}%")