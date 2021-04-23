import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torchvision.tranforms as transforms
from dataset import *
from utils import *
from nltk.translate_bleu_score import corpus_bleu
import torch.nn.functional as F 
from tqdm import tqdm


def evaluate(beam_size):

    # load test data
    