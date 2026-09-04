# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model

# --- Settings ---
EPOCHS = 3          # how many times to go through the full dataset
BATCH_SIZE = 32      # how many images processed at once
LEARNING_RATE = 0.0001  # how big each "nudge" to the weights is


def train():
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data and model
    train_loader, test_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)
    model = get_model(num_classes=2, pretrained=True)
    model = model.to(device)  # move model onto the GPU

    # Loss function: measures how wrong each prediction is
    criterion = nn.CrossEntropyLoss()

    # Optimizer: applies the nudges to the weights, based on the loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()  # tell the model "we're training" (affects some layers' behavior)
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # move data to GPU too

            optimizer.zero_grad()               # clear old nudges from last batch
            outputs = model(images)             # forward pass: get predictions
            loss = criterion(outputs, labels)   # how wrong were we?
            loss.backward()                     # backpropagation: figure out the nudges
            optimizer.step()                    # apply the nudges

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # pick the higher-scoring class as the guess
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} done. Avg loss: {running_loss/len(train_loader):.4f}, Train accuracy: {epoch_acc:.2f}%")

    # Save the trained weights
    torch.save(model.state_dict(), "checkpoints/resnet18_deeptrace.pth")
    print("Model saved to checkpoints/resnet18_deeptrace.pth")


if __name__ == "__main__":
    train()