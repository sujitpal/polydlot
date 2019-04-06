# https://www.tensorflow.org/alpha/guide/eager

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler

class MatrixFactorization(tf.keras.layers.Layer):
    def __init__(self, emb_sz, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        # self.dynamic = True

    def build(self, input_shape):
        num_users, num_movies = input_shape
        self.U = self.add_variable("U", 
            shape=[num_users, self.emb_sz], 
            dtype=tf.float32,
            initializer=tf.initializers.GlorotUniform)
        self.M = self.add_variable("M", 
            shape=[num_movies, self.emb_sz],
            dtype=tf.float32, 
            initializer=tf.initializers.GlorotUniform)
        self.bu = self.add_variable("bu",
            shape=[num_users],
            dtype=tf.float32, 
            initializer=tf.initializers.Zeros)
        self.bm = self.add_variable("bm",
            shape=[num_movies],
            dtype=tf.float32, 
            initializer=tf.initializers.Zeros)
        self.bg = self.add_variable("bg", 
            shape=[],
            dtype=tf.float32,
            initializer=tf.initializers.Zeros)

    def call(self, input):
        return (tf.add(
            tf.add(
                tf.matmul(self.U, tf.transpose(self.M)),
                tf.expand_dims(self.bu, axis=1)),
            tf.expand_dims(self.bm, axis=0)) +
            self.bg)


class MatrixFactorizer(tf.keras.Model):
    def __init__(self, embedding_size):
        super(MatrixFactorizer, self).__init__()
        self.matrixFactorization = MatrixFactorization(embedding_size)
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, input):
        output = self.matrixFactorization(input)
        output = self.sigmoid(output)
        return output


def loss_fn(source, target):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(source, target)
    return loss


def build_numpy_lookup(id2idx, lookup_size):
    lookup = np.zeros((lookup_size))
    idx2id = {idx: id for (id, idx) in id2idx.items()}
    for idx, id in idx2id.items():
        lookup[idx] = id
    return lookup


def load_data():
    movie_id2title = {}
    mid_list = []
    with open(MOVIES_FILE, "r") as fmov:
        for line in fmov:
            if line.startswith("\"movieId"):
                continue
            cols = line.strip().split(",")
            mid, mtitle = cols[0], cols[1]
            mid = int(mid)
            mtitle = mtitle.strip("\"")
            movie_id2title[mid] = mtitle
            mid_list.append(mid)

    unique_uids = set()
    uidmid2ratings = {}
    with open(RATINGS_FILE, "r") as frat:
        for line in frat:
            if line.startswith("\"userId"):
                continue
            cols = line.strip().split(",")
            uid, mid, rating = cols[0], cols[1], cols[2]
            uid = int(uid)
            mid = int(mid)
            rating = float(rating)
            unique_uids.add(uid)
            uidmid2ratings[(uid, mid)] = rating

    uid_list = sorted(list(unique_uids))

    num_users = len(uid_list)
    num_movies = len(mid_list)

    uid2index = {x: i for i, x in enumerate(uid_list)}
    mid2index = {x: i for i, x in enumerate(mid_list)}

    rows, cols, data = [], [], []
    for uid in uid_list:
        for mid in mid_list:
            try:
                data.append(uidmid2ratings[(uid, mid)])
                rows.append(uid2index[uid])
                cols.append(mid2index[mid])
            except KeyError:
                continue

    ratings = csr_matrix((np.array(data),
        (np.array(rows), np.array(cols))),
        shape=(num_users, num_movies), dtype=np.float32)
    scaler = MaxAbsScaler()
    ratings = scaler.fit_transform(ratings)

    X = ratings.todense()

    movie_id2title = {}
    mid_list = []
    with open(MOVIES_FILE, "r") as fmov:
        for line in fmov:
            if line.startswith("\"movieId"):
                continue
            cols = line.strip().split(",")
            mid, mtitle = cols[0], cols[1]
            mid = int(mid)
            mtitle = mtitle.strip("\"")
            movie_id2title[mid] = mtitle
            mid_list.append(mid)

    unique_uids = set()
    uidmid2ratings = {}
    with open(RATINGS_FILE, "r") as frat:
        for line in frat:
            if line.startswith("\"userId"):
                continue
            cols = line.strip().split(",")
            uid, mid, rating = cols[0], cols[1], cols[2]
            uid = int(uid)
            mid = int(mid)
            rating = float(rating)
            unique_uids.add(uid)
            uidmid2ratings[(uid, mid)] = rating

    uid_list = sorted(list(unique_uids))

    rows, cols, data = [], [], []
    for uid in uid_list:
        for mid in mid_list:
            try:
                data.append(uidmid2ratings[(uid, mid)])
                rows.append(uid2index[uid])
                cols.append(mid2index[mid])
            except KeyError:
                continue

    ratings = csr_matrix((np.array(data),
        (np.array(rows), np.array(cols))),
        shape=(num_users, num_movies), dtype=np.float32)
    scaler = MaxAbsScaler()
    ratings = scaler.fit_transform(ratings)

    X = ratings.todense()
    print("X.shape:", X.shape)

    # matrix index to id mappings
    user_idx2id = build_numpy_lookup(uid2index, num_users)
    movie_idx2id = build_numpy_lookup(mid2index, num_movies)

    return X, user_idx2id, movie_idx2id



####################################### main ##########################################

DATA_DIR = Path("../../data")
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
WEIGHTS_FILE = DATA_DIR / "mf-weights.h5"

EMBEDDING_SIZE = 15

BATCH_SIZE = 1
NUM_EPOCHS = 5


X, user_idx2id, movie_idx2id = load_data()

model = MatrixFactorizer(EMBEDDING_SIZE)
model.build(input_shape=X.shape)
model.summary()

optimizer = tf.optimizers.RMSprop(learning_rate=1e-3, momentum=0.9)

losses, steps = [], []
for i in range(1000):
    with tf.GradientTape() as tape:
        Xhat = model(X)
        loss = loss_fn(X, Xhat)
        if i % 100 == 0:
            loss_value = loss.numpy()
            losses.append(loss_value)
            steps.append(i)
            print("step: {:d}, loss: {:.3f}".format(i, loss_value))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

# plot training loss
plt.plot(steps, losses, marker="o")
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()

# save weights from trained model
with h5py.File(WEIGHTS_FILE, "w") as hf:
    for layer in model.layers:
        if layer.name == "matrix_factorization":
            for weight in layer.weights:
                weight_name = weight.name.split("/")[1].split(":")[0]
                weight_value = weight.numpy()
                hf.create_dataset(weight_name, data=weight_value)
    hf.create_dataset("user_idx2id", data=user_idx2id)
    hf.create_dataset("movie_idx2id", data=movie_idx2id)

