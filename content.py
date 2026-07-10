import difflib
import pandas as pd
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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

feature_vectors = vectorizer.fit_transform(cleaned_data["combined_features"])

#print(feature_vectors[1])

similarity = cosine_similarity(feature_vectors, feature_vectors)
#print(similarity[2])
#print(len(similarity))

#SVD
svd = TruncatedSVD(n_components=2)

features_red = svd.fit_transform(feature_vectors)

normalizer = Normalizer()

features_red = normalizer.fit_transform(features_red)

km = KMeans(n_clusters=30)
labels = km.fit_predict(features_red)

print(labels)

# plt.scatter(features_red[:,0], features_red[:,1], c=labels)
# plt.show()

cleaned_data["cluster"] = labels

# for cluster in range(70):
#     print(f"\nCluster {cluster}")
#     print(cleaned_data[cleaned_data.cluster == cluster]["title"].head())


#Heatmap of similarity for cosine similarity
# plt.imshow(similarity, cmap='viridis')
# plt.colorbar()
# plt.show()


list_of_all_titles = cleaned_data['title'].tolist()

# for a single book
def recommend(book_name):

    find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)

    close_match = find_close_match[0]

    index_of_the_book = cleaned_data[cleaned_data.title == close_match].index[0]

    similarity_score = list(enumerate(similarity[index_of_the_book]))

    sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse=True)

    i = 1

    for book in sorted_similar_books:
        index = book[0]
        title_from_index = cleaned_data[cleaned_data.index==index]['title'].values[0]
        
        if(i<= 20):
            print(i, ".", title_from_index)
            i+=1


def recommend_str(books: list[str]):

    favorite_texts = []
    favorite_clusters = []

    for book in books:
        find_close_match = difflib.get_close_matches(book, list_of_all_titles)

        if not find_close_match:

            print(f"No match found for: {book}")

            continue

        close_match = find_close_match[0]

        book_row = cleaned_data[cleaned_data["title"] == close_match]

        text = book_row["combined_features"].values[0]

        favorite_texts.append(text)

    user_profile_text = " ".join(favorite_texts)
    user_profile_vector = vectorizer.transform([user_profile_text])

    similarity_score = cosine_similarity(user_profile_vector, feature_vectors)[0]

    similarity_score = list(enumerate(similarity_score))

    sorted_similar_books = sorted( similarity_score, key=lambda x: x[1], reverse=True)

    i = 1
    cluster_tally = {}

    for book in sorted_similar_books:
        index = book[0]
        title_from_index = cleaned_data[cleaned_data.index==index]['title'].values[0]
        cluster = cleaned_data[cleaned_data.index==index]['categories'].values[0]
        
        if(i<= 40):
            print(i, ".", title_from_index)
            favorite_clusters.append(int(cleaned_data[cleaned_data.index==index]['cluster'].values[0]))

            if cluster in cluster_tally:
                cluster_tally[cluster] += 1
            else:
                cluster_tally[cluster] = 1

            i+=1

    #print(favorite_clusters)
    sorted_cluster_tally = sorted(cluster_tally.items(), key=lambda x: x[1], reverse=True)
    print(sorted_cluster_tally)


        
    
