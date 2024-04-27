import os
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    @property
    def classes(self):
        if not self._classes:
            return list(self.annotations_frame.iloc[:, 1].unique())
        else:
            return list(self._classes)

    @classes.setter
    def classes(self, value):
        self._classes = value
    def __init__(self, dataset=None, root_dir=None, annotations_file=None, transform=None, target_transform=None):
        self.target_transform = target_transform
        self.root_dir = root_dir
        if annotations_file:
            self.classes = None
            self.annotations_frame = self.load_data(annotations_file)
        else:
            self.annotations_frame = self.load_data(root_dir)
        self.transform = transform

    def load_directory(self, path):
        filepaths = list(Path(path).glob(r'**/*.jpg'))
        labels = [str(filepaths[i]).split("\\")[-2] \
                  for i in range(len(filepaths))]
        self.classes = dict.fromkeys(labels)
        filepath = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # Concatenate filepaths and labels
        df = pd.concat([filepath, labels], axis=1)
        df.Label = pd.factorize(df.Label)[0]


        # Shuffle the DataFrame and reset index
        # df = df.sample(frac=1).reset_index(drop=True)

        return df
    def load_path(self, path:str)->pd.DataFrame:
        filename = os.fsdecode(path)
        _, extension = os.path.splitext(filename)
        match extension:
            case '.csv' | 'csv':
                loaded_data = pd.read_csv(path)
            case '.xlsx' | 'xlsx':
                loaded_data = pd.read_excel(path)
            case '':
                loaded_data = self.load_directory(path)
            case _:
                loaded_data = None
        return loaded_data


    def load_data(self, data):
        match data:
            case str():
                loaded_data = self.load_path(data)
            case pd.DataFrame() | torch.utils.data.Dataset():
                loaded_data = data
            case _:
                loaded_data = None

        return loaded_data

    def __getitem__(self, index):
        # row = self.annotations_frame.iloc[index].to_numpy()
        # features = row[1:]
        # label = row[0]
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.annotations_frame.iloc[index, 0]
        if len(img_path.split('/')) == 1:
            img_path = os.path.join(self.root_dir, img_path)
        # print(img_path)
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.annotations_frame.iloc[index, 1]
        # print(label)
        label = int(label)
        # print(image.mode)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        # return features, label

    def __len__(self):
        return len(self.annotations_frame)

