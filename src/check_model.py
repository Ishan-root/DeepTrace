import torch
import torch.nn.functional as F
from dataset import get_dataloaders
from model import get_model


CHECKPOINT_PATH = "checkpoints/resnet18_deeptrace.pth"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Load Sala test set
    _, test_loader, class_names = get_dataloaders()

    print("Class names:", class_names)
    print("Class 0:", class_names[0])
    print("Class 1:", class_names[1])

    # Load trained model
    model = get_model()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)
    )
    model.to(device)
    model.eval()

    print("\nTesting first batch...\n")

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)

    for i in range(10):

        actual = class_names[labels[i].item()]
        predicted_idx = torch.argmax(probs[i]).item()
        predicted = class_names[predicted_idx]

        print(
            f"Image {i}: "
            f"Actual={actual} | "
            f"Predicted={predicted} | "
            f"FAKE={probs[i][0].item()*100:.2f}% | "
            f"REAL={probs[i][1].item()*100:.2f}%"
        )


if __name__ == "__main__":
    main()