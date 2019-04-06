import csv
import h5py
import numpy as np
import operator
import os

from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "../../data"
PICKLED_TENSORS_FILE = os.path.join(DATA_DIR, "mf-weights.h5")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

movie_id2title = {}
with open(MOVIES_FILE, "r") as fmov:
    csv_reader = csv.reader(fmov, delimiter=',')
    next(csv_reader)  # skip header
    for row in csv_reader:
        movie_id = int(row[0])
        movie_title = row[1]
        genres = row[2]
        movie_id2title[movie_id] = (movie_title, genres)

uid2mids = {}
with open(RATINGS_FILE, "r") as frat:
    csv_reader = csv.reader(frat, delimiter=',')
    next(csv_reader)  # skip header
    for row in csv_reader:
        uid, mid, rating = int(row[0]), int(row[1]), float(row[2])
        if uid not in uid2mids.keys():
            uid2mids[uid] = [(mid, rating)] 
        else:
            uid2mids[uid].append((mid, rating))

with h5py.File(PICKLED_TENSORS_FILE, "r") as hf:
    M = hf["."]["M"].value
    U = hf["."]["U"].value
    bg = hf["."]["bg"].value
    bm = hf["."]["bm"].value
    bu = hf["."]["bu"].value
    movie_idx2id = hf["."]["movie_idx2id"].value
    user_idx2id = hf["."]["user_idx2id"].value

# content based: find movies similar to given movie
# Batman & Robin (1997) -- movie_id = 1562
print("*** movies similar to given movie ***")
TOP_N = 10
movie_idx = np.argwhere(movie_idx2id == 1562)[0][0]
source_vec = np.expand_dims(M[movie_idx], axis=1)
movie_sims = np.matmul(M, source_vec)
similar_movie_ids = np.argsort(-movie_sims.reshape(-1,))[0:TOP_N]
for smid in similar_movie_ids:
    movie_id = movie_idx2id[smid]
    title, genres = movie_id2title[movie_id]
    genres = genres.replace('|', ', ')
    print("{:.5f} {:s} ({:s})".format(movie_sims[smid][0], 
        title, genres))

# collaborative filtering based: find movies for user
# user: 121403 has rated 29 movies, we will identify movie
# recommendations for this user that they haven't rated
print("*** top movie recommendations for user ***")
USER_ID = 121403
user_idx = np.argwhere(user_idx2id == USER_ID)
Xhat = (
    np.add(
        np.add(
            np.matmul(U, M.T), 
            np.expand_dims(-bu, axis=1)
        ),
        np.expand_dims(-bm, axis=0)
    ) - bg)
scaler = MinMaxScaler()
Xhat = scaler.fit_transform(Xhat)
Xhat *= 5

user_preds = Xhat[user_idx].reshape(-1)
pred_movie_idxs = np.argsort(-user_preds)

print("**** already rated (top {:d}) ****".format(TOP_N))
mids_already_rated = set([mid for (mid, rating) in uid2mids[USER_ID]])
ordered_mrs = sorted(uid2mids[USER_ID], key=operator.itemgetter(1), reverse=True)
for mid, rating in ordered_mrs[0:TOP_N]:
    title, genres = movie_id2title[mid]
    genres = genres.replace('|', ', ')
    pred_rating = user_preds[np.argwhere(movie_idx2id == mid)[0][0]]
    print("{:.1f} ({:.1f}) {:s} ({:s})".format(rating, pred_rating, title, genres))
print("...")
print("**** movie recommendations ****")
top_recommendations = []
for movie_idx in pred_movie_idxs:
    movie_id = movie_idx2id[movie_idx]
    if movie_id in mids_already_rated:
        continue
    pred_rating = user_preds[movie_idx]
    top_recommendations.append((movie_id, pred_rating))
    if len(top_recommendations) > TOP_N:
        break
for rec_movie_id, pred_rating in top_recommendations:
    title, genres = movie_id2title[rec_movie_id]
    genres = genres.replace('|', ', ')
    print("{:.1f} {:s} ({:s})".format(pred_rating, title, genres))
