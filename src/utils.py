import re
import pandas as pd
import numpy as np
import torch
import cv2
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def clean_document(topic, lsw):
    topic = str(topic)
    t = topic.lower()
    words = re.findall(r'[a-z0-9áéíóúñ]+', t)
    words_ = [w for w in words if w not in lsw and len(w) > 1]
    return str(" ".join(words_))


def clean_price(price):
    price_value = ("".join(re.findall(r'[0-9.]+', price))).replace(',', '')
    return float(price_value)


def generate_dataset(corpus, window_size=5):
    data = []

    for doc in corpus:
        doc = str(doc)
        doc_words = doc.split()
        for i, word in enumerate(doc_words):
            for neighbour in doc_words[max(0, i-window_size):min(len(doc_words), i+window_size+1)]:
                if neighbour != word:
                    data.append((word, neighbour))

    df = pd.DataFrame(data, columns=['word', 'neighbour'])
    return df


def one_hot_encode(word, vocabulary):
    vocab_len = len(vocabulary)
    vector = np.zeros(vocab_len, dtype=int)
    try:
        idx = vocabulary.index(word)
    except:
        idx = vocab_len-1  # word not in vocabulary = UNKNOWN token

    vector[idx] = 1
    return vector


def parseDescriptionVectors(arr):
    elements = []
    for i, e in enumerate(arr):
        elements.append(np.array(e.strip('][').split(', '), dtype=float))
    return np.array(elements)


def batch_train_data(dataset, vocabulary, starting_idx, batch_size):
    words = list(dataset['word'])[starting_idx:starting_idx+batch_size]
    neighbours = list(dataset['neighbour'])[
        starting_idx:starting_idx+batch_size]

    x_train = torch.empty((batch_size, len(vocabulary)))
    y_train = torch.empty((batch_size, len(vocabulary)))

    for i in range(len(words)):
        one_hot_word = torch.tensor(one_hot_encode(words[i], vocabulary))
        one_hot_neighbour = torch.tensor(
            one_hot_encode(neighbours[i], vocabulary))

        x_train[i] = one_hot_word
        y_train[i] = one_hot_neighbour

    return x_train.to(device), y_train.to(device)


def generate_vocabulary_vectors(vocabulary, model):
    vocab_vectors = np.empty((len(vocabulary), len(vocabulary)))

    for i, word in enumerate(vocabulary):
        vocab_vectors[i] = one_hot_encode(word, vocabulary)

    vocab_tensor = torch.tensor(vocab_vectors, dtype=torch.float).to(device)
    with torch.no_grad():
        vocab_vectors = model.vectorize_tokens(vocab_tensor).to('cpu').numpy()
    return vocab_vectors


def evaluate_w2v_model(word_idx, n_samples, vocabulary, vocabulary_tensors):
    distances = list()

    for i in range(len(vocabulary_tensors)):
        distances.append(np.linalg.norm(
            vocabulary_tensors[word_idx]-vocabulary_tensors[i]))

    most_similar_idx = np.argsort(distances)[0:n_samples]
    most_similar_words = [vocabulary[i] for i in most_similar_idx]

    return most_similar_words


"""
Para estandarizacion de datos
"""


def document_to_vector(topic, model):
    words = topic.split()
    document = list()
    for w in words:
        document.append(model.wv[w])
    return np.mean(np.array(document), axis=0)


def z_norm(prices):
    return (prices-np.mean(prices))/np.std(prices)


def encode_targets(category, unique_categories):
    return unique_categories.index(category)


"""
Procesamiento de Imagenes
"""


def resize(image, width, height):
    return cv2.resize(image, (width, height))


def cargaImagenes(rel_path, rutas, width, height):
    images = []
    for ruta in rutas:
        path = rel_path+ruta+".png"
        img = cv2.imread(path)
        if img is not None:
            img = resize(img, width, height)
            img = cv2.normalize(img, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(img)
    images = np.array(images)
    return images


"""
Multimodal helper functions
"""


def multimodal_train_test_split(dataset, split_perc=0.2):
    # Shuffle dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    dataset_length = dataset.shape[0]
    test_data_length = math.ceil(dataset_length*split_perc)

    test_data = dataset.iloc[0:test_data_length].copy().reset_index(drop=True)
    train_data = dataset.iloc[test_data_length:].copy().reset_index(drop=True)

    return train_data, test_data


def get_accuracy(logits, target):
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    return torch.sum(predictions == target).item()
