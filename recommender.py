import content
import pandas as pd
import os

book = input("\nWhat is your favorite book? \n")

if os.path.exists("favorite_books.csv"):
    favorite_books = pd.read_csv("favorite_books.csv")
else:
    favorite_books = pd.DataFrame(columns=["title", "cluster"])

favorite_books.loc[len(favorite_books)] = [book, None]

favorite_books.to_csv("favorite_books.csv", index=False)

print(pd.read_csv("favorite_books.csv")["title"].tolist())

print("\nHere Are My Top 20 Recommendations:\n")

content.recommend_str(favorite_books["title"])

#print("Your favorite genre is cluster ")


