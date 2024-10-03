# [MobileNet V3](https://arxiv.org/abs/1905.02244) in PyTorch

![Downloads](https://img.shields.io/github/downloads/yakhyo/mobilenetv3-pytorch/total) [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/mobilenetv3-pytorch)

MobileNet V3 implementation using PyTorch

**Arxiv:** https://arxiv.org/abs/1905.02244

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Training](#training)
- [Reference](#reference)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from torchvision import transforms

from PIL import Image
from models import mobilenet_v3_small, mobilenet_v3_large
from assets.meta import IMAGENET_CATEGORIES

model = mobilenet_v3_large()
model.load_state_dict("./weights/mobilenet_v3_large.pt")  # weights ported from torchvision
model.float()  # converting weights to float32


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the model's input size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize the image using the mean and std of ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image


def inference(model, image_path):
    model.eval()

    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)

    _, predicted_class = output.max(1)
    print(f"Predicted class index: {predicted_class.item()}")

    predicted_label = IMAGENET_CATEGORIES[predicted_class.item()]
    print(f"Predicted class label: {predicted_label}")


inference(model, "assets/tabby_cat.jpg")
```

## Datasets

ImageNet, folder structure:

```
├── IMAGENET
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ...
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
          ...
```

## Training

Run `main.sh` (for DDP) file by running the following command:

```
bash main.sh
```

`main.sh`:

```
torchrun --nproc_per_node=@num_gpu main.py --epochs 300  --batch-size 512 --lr 0.064  --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2
```

## Reference

- [torchvision](https://github.com/pytorch/vision)
