from pathlib import Path
import pandas as pd

OUT_PATH = Path("books_data/books_cleaned.csv")

books = pd.read_csv(
    "data.csv"
)

#print(books.head())

cleaned = books[[
      "title",
      "subtitle",
      "authors",
      "categories",
      "average_rating",
      "num_pages",
      "ratings_count",
      "published_year"
  ]].copy()

cleaned["combined_features"] = (
      cleaned["title"] +
      cleaned["subtitle"] +
      cleaned["authors"] +
      cleaned["categories"] +
      books["description"]
  )



cleaned = cleaned.dropna(subset=['combined_features'])
print(len(cleaned))

cleaned.to_csv("cleaned.csv", index=False)

#print(cleaned.head())

print(len(cleaned))