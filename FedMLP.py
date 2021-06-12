import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scipy.sparse as sp
import copy
import random


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


# -----------------------------------------------------------------


# --- Functions for MLP model parameter operations ----

def zero_model_parameters(model):
    # sets all parameters to zero

    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data - dict_params2[name1].data)

    model.load_state_dict(dict_params2)


def add_model_parameters(model1, model2):
    # Adds the parameters of model1 to model2

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data + dict_params2[name1].data)

    model2.load_state_dict(dict_params2)


def sub_model_parameters(model1, model2):
    # Subtracts the parameters of model2 with model1

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(dict_params2[name1].data - param1.data)

    model2.load_state_dict(dict_params2)


def divide_model_parameters(model, f):
    # Divides model parameters except for the user embeddings with f
    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 != 'user_embedding.weight':
            dict_params2[name1].data.copy_(param1.data / f)
    model.load_state_dict(dict_params2)


# -----------------------------------------------------------------


# --- CLass definition for the MovieLens data set ---

class RatingsData(Dataset):

    def __init__(self, csv_file, num_negatives=3, validation=True, num_users=6040, num_items=3952, seed=0):
        np.random.seed(seed)
        random.seed(seed)

        self.num_users = num_users
        self.num_items = num_items
        self.current_user = 0

        # Reads the data from file
        r = pd.read_table(csv_file, sep="::",
                          names=["user_id", "movie_id", "rating", "timestamp"], engine='python')

        self.length = len(r)  # Number of interactions in the data set
        self.ratings, self.test = self.load_as_matrix(r,
                                                      validation)  # Interactions as a matrix structured as ((user,item) rating)
        self.num_negatives = num_negatives  # Number of negative instances per positive instance

        # Lists of the users and items to train and test on
        self.pos_user_input, self.pos_item_input, self.pos_rating = [], [], []
        self.neg_user_input, self.neg_item_input, self.neg_rating = [], [], []

        self.user_input, self.item_input, self.rating = [], [], []

        self.get_train_instances()
        self.generate_negatives()

    def __len__(self):
        return len(self.user_input)  # Length of the data to train on

    def __getitem__(self, idx):  # idx is the index of the training instance

        # User and item id as tensors
        user = torch.LongTensor([self.user_input[idx] - 1])  # -1 so that indexing starts from 0
        item = torch.LongTensor([self.item_input[idx] - 1])

        # Output label and loss weight as tensors
        p = torch.ones(1)
        w = torch.ones(1)

        w[0] = 1
        p[0] = min(1, self.rating[idx])

        return user, item, p, w

    def set_current_user(self, u):
        self.current_user = u

    def generate_negatives(self):
        # Samples negative instances for the current user

        num_samples = min(self.num_negatives * len(self.pos_item_input[self.current_user]),
                          self.num_items - len(self.pos_item_input[self.current_user]))
        self.user_input = self.pos_user_input[self.current_user] + [self.current_user + 1] * num_samples
        self.item_input = self.pos_item_input[self.current_user] + random.sample(self.neg_item_input[self.current_user],
                                                                                 num_samples)
        self.rating = self.pos_rating[self.current_user] + [0] * num_samples

    def load_as_matrix(self, ratings, validation):
        # Loads interactions as dictionary of keys (matrix)

        mat = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        last_u = 1
        last_i = 0
        second_last_u = 0
        second_last_i = 0
        test_item = []

        for i in range(self.length):
            user, item, rating = int(ratings["user_id"][i]), int(ratings["movie_id"][i]), int(ratings["rating"][i])

            if user > last_u:
                if validation:  # Use the second last interaction as validation point
                    mat.pop((last_u, last_i))
                    test_item.append(second_last_i)
                    mat.pop((second_last_u, second_last_i))

                else:  # Use last interaction as test point
                    test_item.append(last_i)
                    mat.pop((last_u, last_i))

            if (rating > 0):
                mat[user, item] = rating

            second_last_u = last_u
            second_last_i = last_i
            last_u = user
            last_i = item

        # Last data points in file
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
    def get_train_instances(self):
        train = self.ratings
        user_input, item_input, neg_user, neg_item, labels = [], [], [], [], []
        last_u = 1

        for (u, i) in train.keys():

            if u > last_u:
                self.pos_user_input.append(user_input)
                self.pos_item_input.append(item_input)
                self.pos_rating.append(labels)

                for a in range(1, self.num_items + 1):
                    if a not in item_input:
                        neg_user.append(last_u)
                        neg_item.append(a)

                self.neg_user_input.append(neg_user)
                self.neg_item_input.append(neg_item)

                user_input, item_input, neg_user, neg_item, labels = [], [], [], [], []
                last_u = u

            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(int(train[u, i]))

        self.pos_user_input.append(user_input)
        self.pos_item_input.append(item_input)
        self.pos_rating.append(labels)

        for a in range(1, self.num_items + 1):
            if a not in item_input:
                neg_user.append(last_u)
                neg_item.append(a)

        self.neg_user_input.append(neg_user)
        self.neg_item_input.append(neg_item)


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


# -----------------------------------------------------------------


# --- Function for training FedMLP ---


def fed_fit(model_central, data, C, batch_size, epochs, lr, eta, verbose=True):
    # function for training one round

    device = "cuda"

    # Sample the participants for the round of training
    num_participants = int(data.num_users * C)
    participants = random.sample(range(data.num_users), num_participants)

    # model_difference holds the total change of the global model after the round
    model_difference = copy.deepcopy(model_central)
    zero_model_parameters(model_difference)

    it = 0

    t1 = time.time()

    # Start training loop
    for user in participants:

        it += 1
        if it % int(num_participants / 10) == 0 and verbose:
            print("Progress:", round(100 * it / num_participants), "%")

        # The current user takes a copy of the global model
        model_client = copy.deepcopy(model_central)

        # Defining optimizers
        optimizer = torch.optim.SGD(model_client.parameters(), lr=lr)  # MLP optimizer
        optimizer_u = torch.optim.SGD(model_client.user_embedding.parameters(), lr=lr / C * eta - lr)  # User optimizer
        optimizer_i = torch.optim.SGD(model_client.item_embedding.parameters(),
                                      lr=lr * data.num_items * eta - lr)  # Item optimizer

        # Prepares data for the current user
        data.set_current_user(user)
        data.generate_negatives()

        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=0)

        # Trains on the users data
        for e in range(epochs):
            for batch in dataloader:
                # Load tensors of users, movies, outputs and loss weights
                u, m, p, w = batch
                # move tensors to cuda
                u = u.to(device)
                m = m.to(device)
                p = p.to(device)
                w = w.to(device)

                # make predictions
                p_pred = model_client(u, m)

                # Calculate mean loss
                loss_fn = torch.nn.BCELoss(weight=w, reduction="mean")
                loss = loss_fn(p_pred, p)

                # Backpropagate the output and update model parameters
                optimizer.zero_grad()
                optimizer_u.zero_grad()
                optimizer_i.zero_grad()

                loss.backward()
                optimizer.step()
                optimizer_u.step()
                optimizer_i.step()

        # Calculate the user's change of the model and add it to the total change
        sub_model_parameters(model_central, model_client)
        add_model_parameters(model_client, model_difference)

    # Take the average of the MLP and item vectors
    divide_model_parameters(model_difference, num_participants)

    # Update the global model by adding the total change
    add_model_parameters(model_difference, model_central)
    t2 = time.time()

    print("Time of round:", round(t2 - t1), "seconds")


# -----------------------------------------------------------------


# --- Main script for training---


# Hyper parameters
C = 0.4
E = 1
B = 102
T = 196
lr = 0.3
eta = 80

# Initiate model
model_central = MLP()

print("Processing data...")
data = RatingsData("ratings.dat", num_negatives=3, validation=False)
print("Done")

for t in range(T):  # for each round

    print("Starting round", t + 1)
    # train one round
    fed_fit(model_central, data, C=C, batch_size=B, epochs=E, lr=lr, eta=eta, verbose=True)

    print("Evaluating model...")
    HR, NDCG = evaluate_model(model_central, data, validation=False)
    print("HR@10:", HR)
    print("NDCG@10", NDCG)