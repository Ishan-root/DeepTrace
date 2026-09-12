import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import get_model


IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CHECKPOINT_PATH = "checkpoints/resnet18_deeptrace.pth"


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def diagnose(image_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = get_model()

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    print("\n================================")
    print("EXTERNAL IMAGE DIAGNOSTIC")
    print("================================")

    print("Image:", image_path)
    print("Original size:", image.size)

    print("\nRAW LOGITS")
    print("FAKE:", logits[0][0].item())
    print("REAL:", logits[0][1].item())

    print("\nPROBABILITIES")
    print(
        "FAKE:",
        f"{probabilities[0].item() * 100:.8f}%"
    )

    print(
        "REAL:",
        f"{probabilities[1].item() * 100:.8f}%"
    )

    print("\nLOGIT DIFFERENCE")
    print(
        "REAL - FAKE:",
        logits[0][1].item() - logits[0][0].item()
    )

    print("================================")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python src/diagnose_external.py "
            "path/to/image.jpg"
        )
        sys.exit(1)

    diagnose(sys.argv[1])