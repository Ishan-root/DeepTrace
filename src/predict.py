import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import get_model

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["FAKE", "REAL"]  # matches dataset.py: 0=FAKE, 1=REAL

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def predict(image_path, checkpoint_path="checkpoints/resnet18_deeptrace.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_idx = torch.argmax(probs).item()
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx].item() * 100

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw probabilities -> FAKE: {probs[0].item()*100:.2f}%, REAL: {probs[1].item()*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/your/image.jpg")
        sys.exit(1)

    predict(sys.argv[1])
