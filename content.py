import difflib
import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
plt.imshow(similarity, cmap='viridis')
plt.colorbar()
plt.show()

list_of_all_titles = cleaned_data['title'].tolist()


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

