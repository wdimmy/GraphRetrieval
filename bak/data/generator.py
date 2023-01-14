from tdc.multi_pred import Catalyst
from collections import defaultdict
import pickle, os, random


def generate_pikcle(dataset='USPTO_Catalyst', key1="Reactant", key2="Product"):
    data = Catalyst(name = 'USPTO_Catalyst')
    split = data.get_split()

    train = split["train"]
    valid = split["valid"]
    test = split["test"]

    train_data = defaultdict(list)
    for y, graph_str1, graph_str2 in zip(train["Y"], train[key1], train[key2]):
        train_data[y].append((graph_str1, graph_str2))

    valid_data = defaultdict(list)
    for y, graph_str1, graph_str2 in zip(valid["Y"], valid[key1], valid[key2]):
        valid_data[y].append((graph_str1, graph_str2))

    test_data = defaultdict(list)
    for y, graph_str1, graph_str2 in zip(test["Y"], test[key1], test[key2]):
        test_data[y].append((graph_str1, graph_str2))

    if not os.path.exists(dataset):
        os.mkdir(dataset)

    pickle.dump(train_data, open(os.path.join(dataset, "train.pkl"), "wb"))
    pickle.dump(valid_data, open(os.path.join(dataset, "valid.pkl"), "wb"))
    pickle.dump(test_data,  open(os.path.join(dataset, "test.pkl"), "wb"))


def generate_small_dataset(percentage=0.1, dataset='USPTO_Catalyst'):
    train_data = pickle.load(open(os.path.join(dataset, "train.pkl"), "rb"))
    valid_data = pickle.load(open(os.path.join(dataset, "valid.pkl"), "rb"))
    test_data = pickle.load(open(os.path.join(dataset, "test.pkl"), "rb"))

    train = []
    valid = []
    test = []

    for key, value in train_data.items():
        tmp = random.sample(value, k=int(len(value)*percentage))
        for item in tmp:
            train.append((key, item[0], item[1]))

    for key, value in valid_data.items():
        tmp = random.sample(value, k=int(len(value)*percentage))
        for item in tmp:
            valid.append((key, item[0], item[1]))

    for key, value in test_data.items():
        tmp = random.sample(value, k=int(len(value)*percentage))
        for item in tmp:
            test.append((key, item[0], item[1]))

    print("train=", len(train), "valid=", len(valid), "test=", len(test))
    return train, valid, test


if __name__ == "__main__":
    generate_pikcle()







