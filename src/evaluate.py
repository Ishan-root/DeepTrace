import torch
from dataset import get_dataloaders
from model import get_model

CHECKPOINT_PATH = "checkpoints/resnet18_deeptrace.pth"


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (we only need test_loader here, but get_dataloaders returns both)
    _, test_loader, class_names = get_dataloaders()

    # Load model architecture, then load our trained weights into it
    model = get_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()  # evaluation mode: no gradient updates, no dropout randomness

    correct = 0
    total = 0

    # per-class tracking: correct[0]=FAKE correct, correct[1]=REAL correct, etc.
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():  # no gradients needed, saves memory/speed since we're not training
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")


if __name__ == "__main__":
    evaluate()
