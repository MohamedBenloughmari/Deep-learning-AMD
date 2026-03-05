import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler

import warnings
warnings.filterwarnings("ignore")


class TabularPreprocessor:
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or ['case', 'label', 'LOCALIZER', 'split_type', 'image']
        self.train_columns_ = None
        self.num_cols = None

    def fit_transform(self, df_train):
        tab = df_train.drop(columns=self.drop_cols)
        cat_cols = tab.select_dtypes(include='object').columns.tolist()
        self.num_cols = tab.select_dtypes(exclude='object').columns.tolist()
        tab_encoded = pd.get_dummies(tab, columns=cat_cols, dtype=np.float32)
        self.train_columns_ = tab_encoded.columns
        return tab_encoded

    def transform(self, df):
        tab = df.drop(columns=self.drop_cols)
        cat_cols = tab.select_dtypes(include='object').columns.tolist()
        tab_encoded = pd.get_dummies(tab, columns=cat_cols, dtype=np.float32)
        tab_encoded = tab_encoded.reindex(columns=self.train_columns_, fill_value=0)
        return tab_encoded



class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, mode='both'):
        """
        mode: 'both'      -> returns (image, localiser, label)
              'image'     -> returns (image, label)
              'localiser' -> returns (localiser, label)
              'label'     -> returns label only
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx]['label']

        if self.mode == 'label':
            return label

        if self.mode in ('both', 'image'):
            img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image'])
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

        if self.mode in ('both', 'localiser'):
            localiser_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['LOCALIZER'])
            localiser = Image.open(localiser_name).convert('RGB')
            if self.transform:
                localiser = self.transform(localiser)

        if self.mode == 'image':
            return image, label
        elif self.mode == 'localiser':
            return localiser, label
        else:  # 'both'
            return image, localiser, label




