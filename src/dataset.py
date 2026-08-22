# src/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ResNet18 was originally trained on images resized to 224x224,
# with specific normalization values. We match that here so the
# pretrained weights work well with our data.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_dataloaders(data_dir="data/raw", batch_size=32, num_workers=2):
    """
    Loads train and test datasets from data_dir/train and data_dir/test.
    Expects subfolders REAL/ and FAKE/ inside each.
    Returns: train_loader, test_loader, class_names
    """
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_dataset.classes  # e.g. ['FAKE', 'REAL']
    return train_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check: run this file directly to confirm loading works.
    train_loader, test_loader, class_names = get_dataloaders()
    print("Classes found:", class_names)
    print("Number of training images:", len(train_loader.dataset))
    print("Number of test images:", len(test_loader.dataset))

    # Grab one batch and print its shape, just to confirm everything works
    images, labels = next(iter(train_loader))
    print("One batch of images shape:", images.shape)
    print("One batch of labels:", labels[:10])