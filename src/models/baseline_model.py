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


"""
    Simple Residual Network
"""


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        block_output = self.block(x)
        x = x+block_output
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        self.block1 = Block(in_channels, in_channels)
        self.block2 = Block(in_channels, out_channels)

    def forward(self, x):
        block1_output = self.block1(x)
        x = x+block1_output
        block2_output = self.block2(x)
        x = x+block2_output
        return x


"""
    Custom Multi Modal Network
"""


class MultiModalNeuralNetwork(nn.Module):
    def __init__(self, description_vector_size, description_out_dim, price_out_dim, img_in_channels, img_out_channels, img_height, img_width, product_vector_size, num_classes):
        super(MultiModalNeuralNetwork, self).__init__()

        # Para concatenar las salidas
        self.description_out_dim = description_out_dim
        self.price_out_dim = price_out_dim
        self.img_out_channels = img_out_channels

        self.img_height = img_height
        self.img_width = img_width

        # Arquitectura de la NN
        self.description_forward = DescriptionFeedForward(
            description_vector_size, description_out_dim)
        self.price_forward = PriceFeedForward(price_out_dim)
        self.res_net = ResNet(img_in_channels, img_out_channels)

        self.linear1 = nn.Linear(
            img_height*img_width*img_out_channels + description_out_dim + price_out_dim, 128)
        self.linear2 = nn.Linear(128, product_vector_size)

        # Classifier head
        self.linear3 = nn.Linear(product_vector_size, num_classes)

    def forward(self, description_batch, price_batch, img_batch, embeddings=False):

        description_output = self.description_forward(
            description_batch)  # bs, description_out_dim
        price_output = self.price_forward(price_batch)  # bs, price_out_dim
        img_output = self.res_net(img_batch)  # bs, img_out_channels, h, w

        # flatten img_output
        img_output = img_output.reshape(-1, self.img_out_channels *
                                        self.img_height*self.img_width)

        # Concatenar salida de los 3 modulos
        x = torch.cat([description_output, price_output, img_output], axis=-1)
        x = torch.relu(x)
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

    multimodal_model = MultiModalNeuralNetwork(description_vector_size=VECTOR_SIZE,
                                               description_out_dim=256,
                                               price_out_dim=256,
                                               img_in_channels=3,
                                               img_out_channels=3,
                                               img_height=IMG_HEIGHT,
                                               img_width=IMG_WIDTH,
                                               product_vector_size=300,
                                               num_classes=16)
    print(multimodal_model(desc_input, price_input, img_input).size())
