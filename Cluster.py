import pandas as pd

import difflib
import pandas as pd
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

cleaned_data = pd.read_csv("cleaned.csv")

vectorizer = TfidfVectorizer()

cleaned_data["combined_features"] = (
    cleaned_data["combined_features"]
    .fillna("")
    .astype(str)
)


vectors = vectorizer.fit_transform(cleaned_data["combined_features"])



categories = [] 



# for i in range(2259):
#     if cleaned_data['categories'][i] not in categories:
#         categories.append(cleaned_data['categories'][i])

# # print(categories)
# print(len(categories))

distinct_count = cleaned_data['categories'].unique()

# print(distinct_count)

# svd = TruncatedSVD(n_components=50, random_state=42)
# features_red = svd.fit_transform(vectors)

normalizer = Normalizer()
features_red = normalizer.fit_transform(vectors)

categories = cleaned_data["categories"].dropna().unique()

initial_centers = []

for category in categories:
    # print(f"\nCluster {category}")
    # print(cleaned_data[cleaned_data.categories == category]["title"].head())
    category_books = cleaned_data[cleaned_data.categories == category]

    category_mask = (
          cleaned_data["categories"] == category
      ).to_numpy()

    category_vectors = vectors[category_mask]
    category_features = features_red[category_mask][0].toarray().flatten()
    print(category_features.shape)
    # print(type(category_features))

    similarity = cosine_similarity(category_vectors, vectors)

    rows, columns = similarity.shape

    # print(f"similarity for category: {category}")

    # for row in similarity:
        # print(row)
        # pointwise_average = np.mean(row)
        # print(pointwise_average)

    #  pointwise_average = np.mean(similarity)
    #  print(pointwise_average)

    category_center = np.mean(category_features, axis = 0)
    print(type(category_center))
    initial_centers.append(category_center)
    # print(len(initial_centers))

initial_centers = np.array(initial_centers)

km = KMeans(
    n_clusters=len(initial_centers),
    init=initial_centers,
    n_init=1,
    random_state=42)
labels = km.fit_predict(features_red)

print(labels)

# plt.scatter(features_red[:,0], features_red[:,1], c=labels)
# plt.show()
    
# plt.scatter(
#    features_red[:, 0],
#    features_red[:, 1],
#    c=labels,
#    s=10
#   )

# plt.show()
    

#visualization w/ boundaries

features_2d = features_red[:, :2]
initial_centers_2d = initial_centers[:, :2]

km_2d = KMeans(
    n_clusters=len(initial_centers_2d),
    init=initial_centers_2d,
    n_init=1,
    random_state=42
)

labels_2d = km_2d.fit_predict(features_2d)


x_min = features_2d[:, 0].min() - 0.1
x_max = features_2d[:, 0].max() + 0.1
y_min = features_2d[:, 1].min() - 0.1
y_max = features_2d[:, 1].max() + 0.1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_labels = km_2d.predict(grid_points)
grid_labels = grid_labels.reshape(xx.shape)

plt.figure(figsize=(12, 9))

plt.contourf(
    xx,
    yy,
    grid_labels,
    alpha=0.25,
    cmap="tab20"
)

plt.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    c=labels_2d,
    cmap="tab20",
    s=10
)

plt.scatter(
    km_2d.cluster_centers_[:, 0],
    km_2d.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=100,
    label="Centers"
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Book clusters and decision boundaries")
plt.legend()
plt.show()