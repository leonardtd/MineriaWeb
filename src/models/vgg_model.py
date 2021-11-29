import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Modelo con capas simples custom
"""


def get_accuracy(logits, target):
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    return (torch.sum(predictions == target).item())/float(target.size()[0])


class DescriptionFeedForward(nn.Module):
    def __init__(self, vector_size, output_dim):
        super(DescriptionFeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.feed_forward(x)
        return x


class PriceFeedForward(nn.Module):
    def __init__(self, output_dim):
        super(PriceFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(1, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0, 2),
            nn.Linear(128, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.feed_forward(x)


class MultiModalNeuralNetworkVGG(nn.Module):
    def __init__(self, description_vector_size, description_out_dim, price_out_dim, product_vector_size, num_classes):
        super(MultiModalNeuralNetworkVGG, self).__init__()

        # Para concatenar las salidas
        self.description_out_dim = description_out_dim
        self.price_out_dim = price_out_dim

        # Arquitectura de la NN
        self.description_forward = DescriptionFeedForward(
            description_vector_size, description_out_dim)
        self.price_forward = PriceFeedForward(price_out_dim)

        self.vgg16 = models.vgg16(pretrained=True)

        self.linear1 = nn.Linear(
            1000 + description_out_dim + price_out_dim, 128)
        self.linear2 = nn.Linear(128, product_vector_size)

        # Classifier head
        self.linear3 = nn.Linear(product_vector_size, num_classes)

        # Trainers
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, description_batch, price_batch, img_batch, embeddings=False):

        description_output = self.description_forward(
            description_batch)  # bs, description_out_dim
        price_output = self.price_forward(price_batch)  # bs, price_out_dim
        img_output = self.vgg16(img_batch)  # bs, img_out_channels, h, w

        # flatten img_output
        img_output = img_output.reshape(-1, img_output.shape[1])

        # Concatenar salida de los 3 modulos
        x = torch.cat([description_output, price_output, img_output], axis=-1)
        x = torch.relu(self.linear1(x))

        x_emb = torch.relu(self.linear2(x))
        x = self.linear3(x_emb)

        if embeddings:
            return x, x_emb

        return x  # bs, num_classes

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    VECTOR_SIZE = 300
    IMG_HEIGHT = 244
    IMG_WIDTH = 244
    bs = 128

    desc_input = torch.randn((bs, VECTOR_SIZE))
    price_input = torch.randn((bs, 1))
    img_input = torch.randn((bs, 3, IMG_HEIGHT, IMG_WIDTH))

    multimodal_model = MultiModalNeuralNetworkVGG(description_vector_size=VECTOR_SIZE,
                                                  description_out_dim=256,
                                                  price_out_dim=256,
                                                  product_vector_size=300,
                                                  num_classes=16)
    print(multimodal_model(desc_input, price_input, img_input).size())
