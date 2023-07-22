import os
from PIL import Image
from torch.utils import data


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None):
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples = self.make_dataset(root, self.class_to_idx)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def find_classes(directory):
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        return class_names, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx=None):
        if class_to_idx is None:
            _, class_to_idx = ImageFolder.find_classes(directory)

        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            for root, _, file_names in sorted(os.walk(target_dir, followlinks=True)):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    base, ext = os.path.splitext(path)
                    if ext.lower() in [".jpg", ".jpeg", ".png"]:
                        item = path, class_index
                        instances.append(item)

        return instances
