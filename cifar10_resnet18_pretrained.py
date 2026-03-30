import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():

    # ---------------- CONFIG ---------------- #

    BATCH_SIZE = 64
    EPOCHS = 8
    LR = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # ---------------- DATA ---------------- #

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_ds = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    # ---------------- MODEL ---------------- #

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block + classifier
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    # ---------------- OPTIM ---------------- #

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.3
    )

    # ---------------- TRAIN ---------------- #

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for imgs, labels in train_loader:

            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

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

            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / len(test_ds)
    print("\nTest Accuracy:", acc)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
