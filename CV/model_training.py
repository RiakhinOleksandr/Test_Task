# Importing necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


# Class for our dataset. When initialized transforms values from [0, 1] range to [-1, 1]
class SiameseArrayDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        self.transform = T.Compose([
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]

        # Convert numpy arrays to tensors if needed
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1).float()
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2).float()

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
    

# CNN model that transforms image to embedding vector
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # 1x1
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.25), # 25% random chosen neurons are turned off (outputs setted to zero on forward pass) in training process
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# Our Siamese model which consists of two CNN model (EmbeddingNet)
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, img1, img2):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        return emb1, emb2


# Class for calculating our loss function. If label = 1 (images are matched), it punishes model if distance is too big. If label = 0 (images are not matched), it punishes model if distance is too small
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distances = F.pairwise_distance(emb1, emb2)
        loss = torch.mean(
            label * torch.pow(distances, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        )
        return loss


if __name__ == "__main__":
    # Loading all data from files
    train = np.load("pairs_train_data.npz", allow_pickle=True)
    val = np.load("pairs_validation_data.npz", allow_pickle=True)

    train_pairs = train["pairs"]
    train_labels = train["labels"]
    val_pairs = val["pairs"]
    val_labels = val["labels"]

    train_dataset = SiameseArrayDataset(train_pairs, train_labels)
    val_dataset   = SiameseArrayDataset(val_pairs, val_labels)

    # We will be giving pairs of images by batches for training and for validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # If possible use GPU. It makes training much faster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNet(EmbeddingNet()).to(device) # Setting our Siamese model
    criterion = ContrastiveLoss(margin=1.0) # Setting our loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Setting our optimizer and some important parameters for learning
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5) # Every 10 steps (epochs) learning rate decreases by half
    num_epochs = 50 
    best_val_loss = float("inf") 

    for epoch in range(num_epochs):
        model.train() # Transfer our model into training mode
        train_loss = 0
        for img1, img2, label in train_loader: 
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad() # Resetting gradients, otherwise they accumulate
            emb1, emb2 = model(img1, img2) # Getting embedding vectors of images from CNN
            loss = criterion(emb1, emb2, label) # Calculating loss function
            loss.backward() # Doing backward propogation
            optimizer.step() # Updating weights of model

            train_loss += loss.item()

        model.eval() # Transfer our model into evaluating mode
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                out1, out2 = model(img1, img2) # Getting embedding vectors of images from CNN
                loss = criterion(out1, out2, labels) # Calculating loss function
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        # Check if this is the best result so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_siamese_model.pth")
            print(f"Saved new best model at epoch {epoch+1} (val_loss={val_loss:.4f})")

        scheduler.step() # Doing step for our scheduler (we want to decrease learning rate every 10 steps)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")