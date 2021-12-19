import torch

batch_size = 16
epochs = 50
lr = 0.0001
workers = 0  # Windows下加载数据尽量用0
weights = "./"
image_size = 224
aug_scale = 0.05
aug_angle = 15
filename = '.\checkpoints\checkpoint.pth.tar'

LEARNING_RATE = 1e-4
SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 30
DECAY = 0.5
INTERVAL = 5
NUM_WORKERS = 0
IMAGE_HEIGHT = 1280//2
IMAGE_WIDTH = 1918//2
PIN_MEMORY = True
DATAPATH = "D:\Data\DataSet\carvana-image-masking-challenge"
TRAIN_IMG_DIR = 'D:\Data\DataSet\carvana-image-masking-challenge\\train'
TRAIN_MASK_DIR = 'D:\Data\DataSet\carvana-image-masking-challenge\\train_masks'
TRAIN_TEST_DIR = 'D:\Data\DataSet\carvana-image-masking-challenge\\test'
SUBMIT_DIR = ''