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
from models.baseline_model import MultiModalNeuralNetwork
from models.vgg_model import MultiModalNeuralNetworkVGG


def run_training():

    print("Preparando data...")

    # Read vectorized data
    dfDataset = pd.read_csv(os.path.join(
        config.DATA_DIR, "dataset.csv"), encoding="utf-8")

    train_data, test_data = multimodal_train_test_split(
        dataset=dfDataset, split_perc=0.1)

    # Train data
    train_image_files = train_data["image_name"].apply(lambda x: os.path.join(
        config.DATA_DIR, "imgProyecto/imgProyecto/", x+".png"))

    train_descriptions = parseDescriptionVectors(
        train_data['description_boew'])
    train_prices = train_data['normalized_prices'].tolist()
    train_targets = train_data['target'].tolist()

    # Test data
    test_image_files = test_data["image_name"].apply(lambda x: os.path.join(
        config.DATA_DIR, "imgProyecto/imgProyecto/", x+".png"))
    test_descriptions = parseDescriptionVectors(
        test_data['description_boew'].tolist())
    test_prices = test_data['normalized_prices'].tolist()
    test_targets = test_data['target'].tolist()

    train_dataset = dataset.PlazaVeaDataset(
        image_paths=train_image_files,
        descriptions=train_descriptions,
        prices=train_prices,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_dataset = dataset.PlazaVeaDataset(
        image_paths=test_image_files,
        descriptions=test_descriptions,
        prices=test_prices,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # model config
    print("Configurando Modelo..")

    model = MultiModalNeuralNetworkVGG(description_vector_size=config.VECTOR_SIZE,
                                       description_out_dim=256,
                                       price_out_dim=256,
                                       product_vector_size=300,
                                       num_classes=dfDataset['target'].nunique())
    model.to(config.DEVICE)

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # TensorBoard
    writer = SummaryWriter()
    print("Empezando Entrenamiento..")
    best_acc = 0
    for epoch in range(config.EPOCHS):
        (batch_train_preds, train_loss) = engine.train_fn(
            model, train_loader, cross_entropy_loss, optimizer)

        (batch_preds, test_loss) = engine.eval_fn(
            model, test_loader, cross_entropy_loss
        )

        train_preds = []
        epoch_preds = []

        # Train preds
        for i, vp in enumerate(batch_train_preds):
            train_preds.extend(vp)

        # Test preds
        for i, vp in enumerate(batch_preds):
            epoch_preds.extend(vp)

        #combined_train = list(zip(train_targets, train_preds))
        #combined = list(zip(test_targets, epoch_preds))
        #print("target-pred ", combined[:10])

        train_accuracy = metrics.accuracy_score(train_targets, train_preds)
        test_accuracy = metrics.accuracy_score(test_targets, epoch_preds)
        print(
            f"Epoch={epoch+1}, Train Loss={train_loss}, Train Accuracy={train_accuracy}, Test Loss={test_loss} Test Accuracy={test_accuracy}"
        )

        # Tensorboard stats
        print("Writing to TensorBoard..")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        # Save model
        if test_accuracy > best_acc:
            model.save_model(
                f"model params/model_parameters_vgg_boew")
            best_acc = test_accuracy

    writer.close()


if __name__ == "__main__":
    run_training()
