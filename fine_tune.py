# fine_tune.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
import os
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, text_dir, tokenizer, transform, max_len):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        self.image_paths = []
        self.labels = []
        self.texts = []
        class_folders = sorted(os.listdir(self.image_dir))
        self.label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

        for class_name in class_folders:
            class_path = os.path.join(self.image_dir, class_name)
            if os.path.isdir(class_path):
                file_names = os.listdir(class_path)
                for file_name in file_names:
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(self.label_map[class_name])

                        file_name_no_ext, _ = os.path.splitext(file_name)
                        text = file_name_no_ext.replace('_', ' ')
                        text_without_digits = re.sub(r'\d+', '', text)
                        self.texts.append(text_without_digits)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.transform(Image.open(image_path).convert('RGB'))

        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        label = self.labels[idx]

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. Multimodal Model Definition
class MultimodalClassifier(nn.Module):
    def __init__(self, image_model, text_model, num_classes=4, projection_dim=128):
        super(MultimodalClassifier, self).__init__()

        # Image Model
        self.image_model = image_model
        self.image_model.classifier = nn.Identity()  # Remove classification layer
        image_feature_dim = 1280  # MobileNetV2 feature dimension

        # Text Model
        self.text_model = text_model  # Directly use DistilBertModel
        text_feature_dim = 768  # DistilBERT hidden size

        # Projection layers
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Image features
        img_features = self.image_model(image)
        img_features = self.image_projection(img_features)

        # Text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        txt_features = self.text_projection(cls_output)

        # Concatenate features
        combined_features = torch.cat((img_features, txt_features), dim=1)

        # Classifier
        out = self.classifier(combined_features)
        return out
    
def unfreeze_mobilenet_layers(model, num_layers_to_unfreeze=2):
    """
    Unfreezes the last `num_layers_to_unfreeze` layers of MobileNetV2.
    
    Args:
        model (MultimodalClassifier): The multimodal model containing MobileNetV2.
        num_layers_to_unfreeze (int): Number of layers from the end to unfreeze.
    """
    # MobileNetV2 features are in model.image_model.features
    for child in model.image_model.features[-num_layers_to_unfreeze:]:
        for param in child.parameters():
            param.requires_grad = True

def unfreeze_distilbert_layers(model, num_layers_to_unfreeze=2):
    """
    Unfreezes the last `num_layers_to_unfreeze` transformer layers of DistilBERT.
    
    Args:
        model (MultimodalClassifier): The multimodal model containing DistilBERT.
        num_layers_to_unfreeze (int): Number of transformer layers from the end to unfreeze.
    """
    # DistilBERT has a transformer attribute with 12 transformer layers
    for layer in model.text_model.transformer.layer[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

def main():
    # 1. Data Preparation
    data_dir = r"/work/TALC/enel645_2025w/garbage_data"
    train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
    val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
    test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")
    
    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Rotate images by ±15 degrees
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]),
    ])
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = MultimodalDataset(
        image_dir=train_dir,
        text_dir=train_dir,  # Assuming structured similarly
        tokenizer=tokenizer,
        transform=train_transform,
        max_len=24
    )
    
    val_dataset = MultimodalDataset(
        image_dir=val_dir,
        text_dir=val_dir,
        tokenizer=tokenizer,
        transform=val_test_transform,
        max_len=24
    )
    
    test_dataset = MultimodalDataset(
        image_dir=test_dir,
        text_dir=test_dir,
        tokenizer=tokenizer,
        transform=val_test_transform,
        max_len=24
    )
    
    # Define data loaders
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    # 2. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pre-trained MobileNetV2
    mobile_net = models.mobilenet_v2(pretrained=True)
    for param in mobile_net.parameters():
        param.requires_grad = False
    mobile_net.classifier = nn.Identity()  # Remove classification layer
    
    # Initialize pre-trained DistilBERT
    distil_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    for param in distil_bert.parameters():
        param.requires_grad = False
    
    # Initialize Multimodal Classifier
    model = MultimodalClassifier(
        image_model=mobile_net,
        text_model=distil_bert,
        num_classes=4,
        projection_dim=128
    )
    
    model = model.to(device)
    
    # 3. Load the Best Initial Model Weights
    model.load_state_dict(torch.load("best_multimodal_model.pth"))
    
    # 4. Unfreeze Specific Layers for Fine-Tuning
    unfreeze_mobilenet_layers(model, num_layers_to_unfreeze=2)
    unfreeze_distilbert_layers(model, num_layers_to_unfreeze=2)
    
    # 5. Define Parameter Groups with Differential Learning Rates
    optimizer = optim.Adam([
        {'params': model.image_projection.parameters(), 'lr': 1e-4},
        {'params': model.text_projection.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        # Unfrozen MobileNetV2 layers
        {'params': model.image_model.features[-2:].parameters(), 'lr': 1e-5},
        # Unfrozen DistilBERT transformer layers
        {'params': model.text_model.transformer.layer[-2:].parameters(), 'lr': 1e-5}
    ])
    
    # 6. Define Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # 7. Fine-Tuning Training Loop
    def fine_tune_model(model, dataloaders, criterion, optimizer, device, num_epochs=5):
        best_acc = 0.0
        best_model_wts = None
    
        for epoch in range(num_epochs):
            print(f"Fine-Tuning Epoch {epoch + 1}/{num_epochs}")
            print("-" * 25)
    
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
    
                running_loss = 0.0
                running_corrects = 0
    
                for batch in dataloaders[phase]:
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
    
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images, input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
    
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
    
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
                # Save the best model based on validation accuracy
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, "best_finetuned_multimodal_model.pth")
    
            print()
    
        print(f"Best Validation Accuracy after Fine-Tuning: {best_acc:.4f}")
    
        # Load best model weights
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
    
        return model
    
    # Fine-Tune the Model
    fine_tuned_model = fine_tune_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10
    )
    
    # 8. Evaluation Function
    def evaluate_fine_tuned_model(model, dataloaders, device):
        model.eval()
        running_corrects = 0
        total = 0
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for batch in dataloaders["test"]:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
    
                outputs = model(images, input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
    
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
    
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        test_acc = running_corrects.double() / total
        print(f"Fine-Tuned Test Accuracy: {test_acc * 100:.2f}%")
    
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                    xticklabels=train_dataset.label_map.keys(),
                    yticklabels=train_dataset.label_map.keys())
        plt.title('Confusion Matrix After Fine-Tuning')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
    
    # 9. Evaluate the Fine-Tuned Model
    evaluate_fine_tuned_model(fine_tuned_model, dataloaders, device)

if __name__ == "__main__":
    main()