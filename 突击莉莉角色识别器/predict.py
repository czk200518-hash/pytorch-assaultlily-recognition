import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model import create_model
from dataset_loader import get_transforms


def load_model(model_path: str, device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', [])

    if not class_names:
        classes_path = model_path.replace('.pth', '_classes.json')
        if os.path.exists(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

    args = checkpoint.get('args', {})
    model_name = args.get('model_name', 'standard')
    input_size = args.get('input_size', 128)
    num_classes = len(class_names)

    if num_classes == 0:
        raise ValueError('无法确定类别数量，请确保模型文件包含 class_names')

    model = create_model(model_name, num_classes, input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(model, image_path, class_names, device, input_size=128):
    transform = get_transforms(input_size, is_train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence = max_prob.item()

    top3_prob, top3_idx = torch.topk(probabilities, min(3, len(class_names)))
    top3_results = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3': top3_results,
    }


def predict_folder(model, folder_path, class_names, device, input_size=128):
    results = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    folder = Path(folder_path)
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            result = predict_image(
                model, str(img_path), class_names, device, input_size
            )
            results[img_path.name] = result

    return results


def extract_feature(model, image_path, device, input_size=128):
    transform = get_transforms(input_size, is_train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(image_tensor)

    return features.cpu().numpy().flatten()
