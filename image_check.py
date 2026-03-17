import os
import imagehash
import torch
from PIL import Image
from torchvision import transforms, models

IMAGE_FOLDER = "uploaded_images"

# Load trained AI model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("ai_image_detector.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def check_duplicate(image_path):
    new_hash = imagehash.phash(Image.open(image_path))

    for file in os.listdir(IMAGE_FOLDER):
        existing_path = os.path.join(IMAGE_FOLDER, file)

        if existing_path == image_path:
            continue

        existing_hash = imagehash.phash(Image.open(existing_path))

        if new_hash - existing_hash < 5:
            return True

    return False


def detect_ai_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return prediction