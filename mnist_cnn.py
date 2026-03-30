import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# ---------------- CONFIG ---------------- #

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- DATA ---------------- #

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

test_ds = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL ---------------- #

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 🔥 Learning-rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.3
)

# ---------------- TRAIN ---------------- #

train_losses = []

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for imgs, labels in train_loader:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

# ---------------- TEST ---------------- #

model.eval()

correct = 0
all_preds = []
all_labels = []

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = correct / len(test_ds)
print("\nTest Accuracy:", acc)

# ---------------- CONFUSION MATRIX ---------------- #

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

# ---------------- MISCLASSIFIED DIGITS ---------------- #

wrong = []

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)

        for i in range(len(preds)):
            if preds[i].cpu() != labels[i]:
                wrong.append((imgs[i].cpu(), labels[i], preds[i]))
            if len(wrong) >= 9:
                break

        if len(wrong) >= 9:
            break


plt.figure(figsize=(6, 6))

for i, (img, t, p) in enumerate(wrong):

    plt.subplot(3, 3, i + 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"T:{t}  P:{p}")
    plt.axis("off")

plt.suptitle("Misclassified Digits")
plt.show()
