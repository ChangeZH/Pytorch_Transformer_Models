import os
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Classification_Datasets(Dataset):
    """docstring for Classification_Datasets"""

    def __init__(self, set_name, root_path, num_classes, input_size, input_channel, **kwargs):
        super(Classification_Datasets, self).__init__()
        self.set_name = set_name
        self.input_channel = input_channel

        img_list = {class_name: os.listdir(os.path.join(root_path, self.set_name, class_name))
                    for class_name in os.listdir(os.path.join(root_path, self.set_name))}
        img_num = {class_name: len(img_list[class_name]) for class_name in img_list}
        self.labels_id = {class_name: [_ for _ in img_list].index(class_name) for class_name in img_list}
        self.id_labels = {self.labels_id[class_name]: class_name for class_name in self.labels_id}
        labels = {class_name: [self.labels_id[class_name]] * img_num[class_name] for class_name in img_list}
        self.img_path = []
        self.img_label = []
        for class_name in img_list:
            self.img_path += [os.path.join(root_path, set_name, class_name, _) for _ in img_list[class_name]]
            self.img_label += labels[class_name]
        self.transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('L') \
            if self.input_channel == 1 else Image.open(self.img_path[item]).convert('RGB')
        return self.transform(img), self.img_label[item]
