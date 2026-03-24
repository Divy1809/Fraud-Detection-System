from datasets import load_dataset
import torch
from torchvision import transforms, models
from torch import nn
from torch.utils.data import DataLoader

# 🔹 load dataset (keep small for now)
dataset = load_dataset(
    "Hemg/AI-Generated-vs-Real-Images-Datasets",
    split="train[:3000]"
)

# 🔹 transform (VERY IMPORTANT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(   # ✅ REQUIRED
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(example):
    image = example["image"].convert("RGB")
    example["pixel_values"] = transform(image)
    return example

dataset = dataset.map(preprocess)
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# 🔹 dataloader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 🔹 USE RESNET18 (FASTER)
model = models.resnet18(weights="IMAGENET1K_V1")

# 🔹 replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

device = "cpu"
model.to(device)

# 🔹 loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 🔹 training (LESS EPOCHS)
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# 🔹 save model
torch.save(model.state_dict(), "ai_model.pth")

print("✅ Model trained and saved")