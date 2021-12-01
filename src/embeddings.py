import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import engine
from utils import *


# Models
from models.vgg_model import MultiModalNeuralNetworkVGG
from models.resnet_model import MultiModalNeuralNetworkResNet
from models.best_model import MultiModalNeuralNetwork

"""
    Script para generar embeddings segun un modelo
"""


def get_embeddings():

    print("Preparando data...")

    # Read vectorized data
    test_data = pd.read_csv(os.path.join(
        config.DATA_DIR, "dataset.csv"), encoding="utf-8")

    # Test data
    test_image_files = test_data["image_name"].apply(lambda x: os.path.join(
        config.DATA_DIR, "imgProyecto/imgProyecto", x+".png"))

    test_descriptions = parseDescriptionVectors(
        test_data['description_boew'].tolist())
    test_prices = test_data['normalized_prices'].tolist()
    test_targets = test_data['target'].tolist()

    test_dataset = dataset.PlazaVeaDataset(
        image_paths=test_image_files,
        descriptions=test_descriptions,
        prices=test_prices,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # model config
    print("Configurando Modelo..")

    model = MultiModalNeuralNetworkResNet(description_vector_size=config.VECTOR_SIZE,
                                          description_out_dim=256,
                                          price_out_dim=256,
                                          product_vector_size=300,
                                          num_classes=16)

    # Load params
    model.load_model("./model params/model_parameters_resnet_boew")

    model.to(config.DEVICE)

    print("Empezando Procesamiento..")
    best_acc = 0

    batch_embeddings = engine.emb_fn(
        model, test_loader)

    final_embeddings = []

    # Test preds
    for i, vp in enumerate(batch_embeddings):
        final_embeddings.extend(vp)

    test_data["resnet_embeddings"] = final_embeddings
    test_data.to_csv("resnet_productos.csv",
                     encoding="utf-8", index=False)


if __name__ == "__main__":
    get_embeddings()
