# src/model.py

import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2, pretrained=True):
    """
    Loads ResNet18, pretrained on ImageNet, and replaces the final
    layer so it outputs predictions for our 2 classes (FAKE/REAL)
    instead of the original 1000 ImageNet classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # ResNet18's final layer is called 'fc' (fully connected).
    # It currently maps features to 1000 classes. We replace it
    # with a fresh layer mapping features to our 2 classes.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and pass one dummy image through it.
    model = get_model()
    print(model.fc)  # confirm final layer now outputs 2 classes

    dummy_input = torch.randn(1, 3, 224, 224)  # fake image: batch of 1
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [1, 2]
    print("Raw output values:", output)