import torch
from torch.utils.data import Dataset
import torchvision.transforms as transfroms

from PIL import Image
import pandas as pd
import spacy

# set up spacy for English
spacy_en = spacy.load('en')


class CaptionDataset(Dataset):
    """
    Caption Dataset Class
    """

    def __init__(self, imgs_dir, captions_file, vocab, transforms=None, split='train'):
        """
        :param imgs_dir: folder where images are stored
        :param captions_file: the df file with all caption information
        :param vocab: vocabuary object
        :param transforms: image transforms pipeline
        :param split: data split
        """

        # split has to be one of {'train', 'val', 'test'}
        assert split in {'train', 'val', 'test'}

        self.imgs_dir = imgs_dir
        self.df = pd.read_csv(captions_file)
        self.df = self.df[self.df['split'] == split]
        self.vocab = vocab
        self.transforms = transforms
        self.split = split

        self.dataset_size = self.df.shape[0]
        # printing some info
        print(f"Dataset split: {split}")
        print(f"Unique images: {self.df.file_name.nunique()}")
        print(f"Total size: {self.dataset_size}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        # loading the image
        img_id = self.df['file_name'].values[index]
        img = transfroms.ToTensor()(Image.open(
            self.imgs_dir+img_id).convert("RGB"))
        if self.transforms is not None:
            img = self.transforms(img)

        # loading current caption
        cap_len = self.df['tok_len'].values[index]
        caption = self.df['caption'].values[index]
        caption = torch.LongTensor(self.vocab.numericalize(caption))

        if self.split is 'train':
            return img, caption, cap_len
        else:
            # for val and test return all captions for calculate the bleu scores
            captions_text = self.df[self.df['file_name']
                                    == img_id].caption.values
            all_captions = []
            for cap in captions_text:
                all_captions.append(self.vocab.numericalize(cap))

            return img, caption, cap_len, all_captions
