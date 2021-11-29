import torch

DATA_DIR = "../dataset/"
BATCH_SIZE = 16
IMAGE_WIDTH = 244
IMAGE_HEIGHT = 244
VECTOR_SIZE = 300
NUM_WORKERS = 8
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
