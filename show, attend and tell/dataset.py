import torch
from torch.data.utils import Dataset
import torchvision.transforms as transfroms

from PIL import Image
import pandas as pd
import spacy

# set up spacy for English
spacy_en = spacy.load('en')


class CaptionDataset(Dataset):

    def __init__(self, imgs_dir, captions_file, vocab, transforms=None, freq=2, split='train'):
        # split has to be one of {'train', 'val', 'test'}
        assert split in {'train', 'val', 'test'}

        self.imgs_dir = imgs_dir
        self.df = pd.read_csv(captions_file)
        self.vocab = vocab
        self.transforms = transforms

        # printing some info
        print(f"Dataset split: {split}")
        print(f"Unique images: {self.df.file_name.nunique()}")
        print(f"Total size: {self.df.shape[0]}")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        # loading the image
        img_id = self.df['file_name'].values[index]
        img = torch.FloatTensor(Image.open(self.imgs_dir+img_id).covert("RGB"))
        if self.transforms is not None:
            img = self.transforms(img)

        # loading current caption
        word_idx = self.vocab.numericalize(caption)
