import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery

app = Flask(__name__)

credentials_path = "/app/keys/direct-hope-387806-8d2b15781824.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = bigquery.Client()

def load_data_from_bigquery(table_name, selected_columns=None):
    table_ref = client.dataset('Jejaka').table(table_name)
    table = client.get_table(table_ref)
    
    if selected_columns is None:
        data = client.list_rows(table).to_dataframe()
    else:
        schema = table.schema
        selected_fields = [field for field in schema if field.name in selected_columns]
        data = client.list_rows(table, selected_fields=selected_fields).to_dataframe()
    
    return data

item_data_tourism = load_data_from_bigquery('item_data_tourism', ['place_id', 'place_name', 'desc', 'place_address', 'total_review', 'ave_rating'])
item_data_restaurant = load_data_from_bigquery('item_data_restaurant', ['place_id', 'place_name', 'desc', 'place_address', 'total_review', 'ave_rating'])
item_data_hotel = load_data_from_bigquery('item_data_hotel', ['place_id', 'place_name', 'desc', 'place_address', 'total_review', 'ave_rating'])

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    input_text = data.get('input_text', '')
    filters = data.get('filters', [])

    if 'tourism' in filters:
        item_data = item_data_tourism
    if 'restaurant' in filters:
        item_data = item_data_restaurant
    if 'hotel' in filters:
        item_data = item_data_hotel

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(item_data['place_name'] + ' ' + item_data['desc'] + ' ' + item_data['place_address'])

    tfidf_matrix_input = tfidf_vectorizer.transform([input_text])

    similarity_scores = cosine_similarity(tfidf_matrix_input, tfidf_matrix)
    similarity_scores = similarity_scores.flatten()
    top_k_indices = np.argsort(similarity_scores)[::-1][:20]
    top_k_places = item_data.iloc[top_k_indices]

    results = []
    for index, row in top_k_places.iterrows():
        result = {
            'place_id': row['place_id'],
            'place_name': row['place_name'],
            'place_address': row['place_address'],
            'total_review': row['total_review'],
            'ave_rating': row['ave_rating']
        }
        results.append(result)

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
