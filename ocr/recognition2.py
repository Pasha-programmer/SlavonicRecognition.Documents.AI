#!/usr/bin/env python3
"""
Программа для распознавания символа глаголицы на изображении.
Использует обученную модель из файла glagolitic_model_full.pth.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import sys


def get_model(num_classes: int) -> nn.Module:
    """
    Создаёт архитектуру модели, совместимую с сохранённой.
    Используется ResNet18 с заменой последнего полносвязного слоя.
    """
    model = models.resnet18(weights=None)  # веса загрузим позже из чекпоинта
    # Заменяем классификатор: Dropout + Linear
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model


def load_model(checkpoint_path: str, device: torch.device):
    """
    Загружает модель и мета-информацию из файла чекпоинта.
    Возвращает (model, idx_to_label, transform, device)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Файл модели не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Извлекаем параметры
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    mean = checkpoint['mean']
    std = checkpoint['std']
    idx_to_label = checkpoint['idx_to_label']  # словарь {index: label}
    label_to_idx = checkpoint['label_to_idx']  # может пригодиться, но не обязательно

    # Создаём модель и загружаем веса
    model = get_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Трансформации для входного изображения (должны совпадать с валидационными)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return model, idx_to_label, transform


def predict_image(model, image_path: str, transform, device, idx_to_label, top_k: int = 1):
    """
    Классифицирует одно изображение.
    Возвращает список (класс, вероятность) для top_k предсказаний.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    # Загружаем и преобразуем изображение
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки изображения {image_path}: {e}")

    img_tensor = transform(img).unsqueeze(0).to(device)  # добавляем batch dimension

    # Инференс
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    # Получаем top-k вероятностей и индексы
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    results = []
    for i, idx in enumerate(top_indices):
        label = idx_to_label[str(idx)] if isinstance(idx_to_label, dict) else idx_to_label[idx]
        results.append((label, top_probs[i]))

    return results

def start_recognition2(args: object):
    """Начать распознавание. args: { image: str, model: str, top_k: int, useCpu: bool }"""
    
    # parser = argparse.ArgumentParser(description='Распознавание символов глаголицы на изображении')
    # parser.add_argument('image', type=str, help='Путь к файлу изображения')
    # parser.add_argument('--model', type=str, default='glagolitic_model_full.pth',
    #                     help='Путь к файлу модели (по умолчанию: glagolitic_model_full.pth)')
    # parser.add_argument('--top_k', type=int, default=1,
    #                     help='Выводить top K наиболее вероятных классов (по умолчанию: 1)')
    # parser.add_argument('--cpu', action='store_true',
    #                     help='Принудительно использовать CPU (даже если есть GPU)')
    # args = parser.parse_args()

    # Определяем устройство
    if args.useCpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Загружаем модель
    print("Загрузка модели...")
    try:
        model, idx_to_label, transform = load_model(args.model, device)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}", file=sys.stderr)
        sys.exit(1)

    # Распознаём изображение
    print(f"Анализ изображения: {args.image}")
    try:
        predictions = predict_image(model, args.image, transform, device, idx_to_label, top_k=args.top_k)
    except Exception as e:
        print(f"Ошибка при распознавании: {e}", file=sys.stderr)
        sys.exit(1)

    # Выводим результаты
    print("\nРезультаты распознавания:")
    for i, (label, prob) in enumerate(predictions):
        print(f"{i+1}. Символ: {label} (вероятность: {prob:.4%})")


if __name__ == "__main__":
    main()