from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.pydantic import Vector, LanceModel

import numpy as np
import pandas as pd

class BookTitle(LanceModel):
    vector: Vector(384)
    title: str

class VectorDatabase:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.client = lancedb.connect(self.db_path)
        self.table_name = table_name
        self.table = self.client.create_table(self.table_name, schema=BookTitle.to_arrow_schema())


    def ingest_data(self, title_data):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = model.encode(title_data)
        embeddings = np.array(embeddings, dtype=np.float32)

        embeddings = model.encode(title_data, convert_to_tensor=True).tolist()
        db_titles_data = []
        for title, embedding in zip(title_data, embeddings):
            embedding = embedding / np.linalg.norm(embedding)
            db_titles_data.append(BookTitle(title= title, vector= embedding))
        self.table.add(db_titles_data)

    def query(self, query_title, top_k=5):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query_vector = model.encode([query_title])
        results = self.table.search(query_vector).limit(top_k).to_list()
        similar_titles = [r['title'] for r in results]

        return similar_titles


if __name__ == "__main__":
    titles = pd.read_csv("titles.csv")["Title"]
    db_titles = titles[1:]

    db_path = "./book_titles"
    table_name = "book_titles"
    vector_db = VectorDatabase(db_path, table_name)

    vector_db.ingest_data(titles)

    query = titles[0]
    print("Query:")
    print(query)
    print("---")
    results = vector_db.query(query, top_k=5)

    print("Query results:")
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result}")
