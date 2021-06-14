import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scipy.sparse as sp


# --- Class definition of the MLP model ---

class MLP(nn.Module):
    ''' Constructs a Multi-Layer Perceptron model'''

    def __init__(self, num_users=6040, num_items=3952, embeddings=64):
        torch.manual_seed(0)
        super().__init__()

        # user and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embeddings).cuda()
        self.item_embedding = nn.Embedding(num_items, embeddings).cuda()

        # MLP layers
        self.l1 = nn.Linear(embeddings * 2, 64).cuda()
        self.l2 = nn.Linear(64, 32).cuda()
        self.l3 = nn.Linear(32, 16).cuda()
        self.l4 = nn.Linear(16, 1, bias=False).cuda()

    def forward(self, user, item):
        # map to embeddings
        embedding1 = self.user_embedding(user).squeeze(1)
        embedding2 = self.item_embedding(item).squeeze(1)

        # Concatenation of the embedding layers
        out = torch.cat((embedding1, embedding2), 1)

        # feed through the MLP layers
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        # output between 0 and 1
        out = torch.sigmoid(self.l4(out))
        return out


# --- Class definition of the GMF model ---

class GMF(nn.Module):
    ''' Constructs a Generalized Matrix Factorization model '''

    def __init__(self, num_users=6040, num_items=3952, embeddings=64):
        torch.manual_seed(0)
        super().__init__()

        # user and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embeddings).cuda()
        self.item_embedding = nn.Embedding(num_items, embeddings).cuda()

    def forward(self, user, item):
        # map to embeddings
        embedding1 = self.user_embedding(user).squeeze(1)
        embedding2 = self.item_embedding(item).squeeze(1)

        # Elementwise multiplication
        GMF_layer = embedding1 * embedding2

        # sum GMF layer
        out = torch.sum(GMF_layer, 1).unsqueeze_(1)

        # output between 0 and 1
        out = torch.sigmoid(out)

        return out


# -----------------------------------------------------------------


# --- CLass definition for the MovieLens data set ---

class RatingsData(Dataset):

    def __init__(self, csv_file, num_negatives=3, validation=True, num_users=6040, num_items=3952):
        np.random.seed(0)
        self.num_users = num_users
        self.num_items = num_items
        # Reads the data from file
        r = pd.read_table(csv_file, sep="::",
                          names=["user_id", "movie_id", "rating", "timestamp"], engine='python')

        self.length = len(r)  # Number of interactions in the data set
        self.ratings, self.test = self.load_as_matrix(r,
                                                      validation)  # Interactions as a matrix structured as ((user,item) rating)
        self.num_negatives = num_negatives  # Number of negative instances per positive instance
        # Lists of the users and items to train and test on
        self.user_input, self.item_input, self.rating = [], [], []
        self.get_train_instances(0)

    def __len__(self):
        return len(self.user_input)  # Length of the data to train on

    def __getitem__(self, idx):  # idx is the index of the training instance
        # User and item id as tensors
        user = torch.LongTensor([self.user_input[idx] - 1])  # -1 so that indexing starts from 0
        movie = torch.LongTensor([self.item_input[idx] - 1])
        # Output label and loss weight as tensors
        p = torch.ones(1)
        w = torch.ones(1)
        # Larger weight for higher ratings
        w[0] = 1
        p[0] = min(1, self.rating[idx])

        return user, movie, p, w

    def load_as_matrix(self, ratings, validation):
        # Interactions as dictionary of keys (matrix)
        mat = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        last_u = 1
        last_i = 0
        second_last_u = 0
        second_last_i = 0
        test_item = []

        for i in range(self.length):
            user, item, rating = int(ratings["user_id"][i]), int(ratings["movie_id"][i]), int(ratings["rating"][i])
            if user > last_u:
                if validation:
                    mat.pop((last_u, last_i))
                    test_item.append(second_last_i)
                    mat.pop((second_last_u, second_last_i))
                else:
                    test_item.append(last_i)
                    mat.pop((last_u, last_i))

            if (rating > 0):
                # mat[user, item] = 1.0
                mat[user, item] = rating

            second_last_u = last_u
            second_last_i = last_i
            last_u = user
            last_i = item

        if validation:
            mat.pop((last_u, last_i))
            test_item.append(second_last_i)
            mat.pop((second_last_u, second_last_i))
        else:
            test_item.append(last_i)
            mat.pop((last_u, last_i))

        return mat, test_item

    # if validation is True the last two interaction for each user will not be
    # part of the training instances, and the penultimate will be used as test
    # if False only the last interaction for each user will be left out and used as test
    def get_train_instances(self, seed=0):
        np.random.seed(seed)
        train = self.ratings
        self.user_input, self.item_input, self.rating = [], [], []

        for (u, i) in train.keys():

            # positive instance
            self.user_input.append(u)
            self.item_input.append(i)
            self.rating.append(int(train[u, i]))

            # Generate negative instances
            for t in range(self.num_negatives):
                j = np.random.randint(1, self.num_items + 1)
                # Keep generating items if the item has been interacted with
                while (u, j) in train:
                    j = np.random.randint(1, self.num_items + 1)
                self.user_input.append(u)
                self.item_input.append(j)
                self.rating.append(0)


# -----------------------------------------------------------------


# --- Functions for testing ---

def test_negatives(filename):
    # Read items (from file) that each user has not interacted with and add them to a list

    with open(filename, "r") as f:
        line = f.readline()
        negatives = []
        while line != None and line != "":
            arr = line.split("\t")
            for x in arr:
                if x != "\n":
                    negatives.append(int(x))
            line = f.readline()

    return negatives


def rank(l, item):
    # rank of the test item in the list of negative instances
    # returns the number of elements that the test item is bigger than

    index = 0
    for element in l:
        if element > item:
            index += 1
            return index
        index += 1
    return index


def get_test_tensor(user, test_item, test_neg):
    # Prepare the test data points as tensors

    test_item = test_item[user - 1]

    test_negatives = test_neg[100 * (user - 1):100 * user]

    user_tensor = torch.LongTensor([user - 1])
    user_test = torch.stack((user_tensor, user_tensor))
    item_input = test_negatives[0] - 1
    item_input = torch.LongTensor([item_input])

    user_tensor.unsqueeze_(0)

    item_test = torch.LongTensor([test_item - 1])
    item_test = torch.stack((item_test, item_input))

    for i in range(1, 100):
        item_input = test_negatives[i] - 1
        item_input = torch.LongTensor([item_input])
        item_input.unsqueeze_(0)

        user_test = torch.cat((user_test, user_tensor), 0)
        item_test = torch.cat((item_test, item_input), 0)

    return user_test, item_test


def evaluate_model(model, data, validation=True, num_users=6040):
    # Evaluates the model and returns HR@10 and NDCG@10

    device = "cuda"
    test_items = data.test
    if validation:
        test_neg = test_negatives("validation_negatives.csv")
    else:
        test_neg = test_negatives("test_negatives.csv")
    hits = 0
    ndcg = 0

    for i in range(1, num_users + 1):
        user_test, item_test = get_test_tensor(i, test_items, test_neg)

        user_test = user_test.to(device)
        item_test = item_test.to(device)

        l = model(user_test, item_test)
        l = l.tolist()
        l = sum(l, [])
        first = l.pop(0)

        l.sort()

        ranking = rank(l, first)

        if ranking > 90:
            hits += 1
            ndcg += np.log(2) / np.log(len(user_test) - ranking + 1)

    hr = hits / data.num_users
    ndcg = ndcg / data.num_users
    return hr, ndcg


# --- Function for training ---

def fit(model, data, batch_size, epochs, lr, wd, validation=True, verbose=True):
    device = "cuda"
    # Defining optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    data_length = len(data)
    it_per_epoch = len(data) / batch_size
    tot_loss = 0

    t1 = time.time()

    # Start training loop
    for e in range(epochs):
        print("Starting epoch ", e + 1)
        data.get_train_instances(seed=e)

        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        t1 = time.time()
        i = 0
        for batch in dataloader:
            # Load tensors of users, movies, outputs and loss weights
            u, m, p, w = batch
            # move tensors to cuda
            u = u.to(device)
            m = m.to(device)
            p = p.to(device)
            w = w.to(device)

            # make predictions
            p_pred = model(u, m)
            # Calculate mean loss
            loss_fn = torch.nn.BCELoss(weight=w, reduction="mean")
            loss = loss_fn(p_pred, p)
            tot_loss += loss.item()

            # Backpropagate the output and updates model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

            # Print progress
            if i % int(1 + it_per_epoch / 10) == 0 and verbose:
                print("Progress: ", round(100 * i / it_per_epoch), "%")

        # Epoch metrics
        t2 = time.time()
        print("Epoch time:", round(t2 - t1), "seconds")
        print("Loss:", tot_loss / i)
        print("Evaluating model...")
        HR, NDCG = evaluate_model(model, data, validation=validation)
        print("HR@10:", HR)
        print("NDCG@10", NDCG)
        tot_loss = 0
        print()

    print("Done")


# -----------------------------------------------------------------


# --- Main script for training---

print("Processing data...")
data = RatingsData("ratings.dat", num_negatives=3, validation=False)
print("Done")

learning_rate = pow(10, -2)
weight_decay = 0

model = MLP()
# for GMF: model = GMF()

# train and test
fit(model=model, data=data, batch_size=256, epochs=10, lr=learning_rate, wd=weight_decay, validation=False,
    verbose=True)