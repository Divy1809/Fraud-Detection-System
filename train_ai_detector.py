from datasets import load_dataset
import torch
from torchvision import transforms, models
from torch import nn
from torch.utils.data import DataLoader
import os

# device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load dataset
dataset = load_dataset(
    "Hemg/AI-Generated-vs-Real-Images-Datasets",
    split="train[:6000]"
)

# image transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def preprocess(example):
    image = example["image"].convert("RGB")
    example["pixel_values"] = transform(image)
    return example

dataset = dataset.map(preprocess)
dataset.set_format(type="torch", columns=["pixel_values","label"])

# dataloader (smaller batch for laptop)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# load pretrained model
model = models.mobilenet_v2(weights="DEFAULT")

# modify final layer for 2 classes
model.classifier[1] = nn.Linear(model.last_channel, 2)

# move model to device
model = model.to(device)

# LOAD PREVIOUS MODEL IF EXISTS
model_path = "ai_image_detector.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Previous trained model loaded. Resuming training...")
else:
    print("No previous model found. Training from scratch...")

# loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# training loop
epochs = 3

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in train_loader:
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# save model
torch.save(model.state_dict(), model_path)

print("Model saved successfully.")
