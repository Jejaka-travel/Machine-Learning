import os
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

app = Flask(__name__)

credentials_path = "/app/keys/direct-hope-387806-8d2b15781824.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = bigquery.Client()

def load_target_data(place_id):
    table_ref = client.dataset('Jejaka').table('target_data_hotel')
    try:
        table = client.get_table(table_ref)
        query = f"SELECT * FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}` WHERE place_id = '{place_id}'"
        data = client.query(query).to_dataframe()
        return data
    except NotFound:
        return pd.DataFrame()

def add_review_to_bigquery(user_id, place_id, user_review, user_rating):
    table_ref = client.dataset('Jejaka').table('target_data_hotel')
    table = client.get_table(table_ref)
    schema = table.schema

    rows = [{'user_id': user_id, 'place_id': place_id, 'user_review': user_review, 'user_rating': user_rating}]
    errors = client.insert_rows(table_ref, rows, selected_fields=schema)

    if errors:
        raise ValueError(f"Failed to insert rows: {errors}")

@app.route('/review-hotel', methods=["POST"])
def get_review():
    data = request.get_json()

    user_id = data.get('user_id', '')
    place_id = data.get('place_id', '')

    target_data = load_target_data(place_id)
    filtered_data = target_data[target_data['place_id'] == place_id]

    result = []
    if not filtered_data.empty:
        for index, row in filtered_data.iterrows():
            result.append({
                'user_id': row['user_id'],
                'place_id': row['place_id'],
                'user_review': row['user_review'],
                'user_rating': row['user_rating']
            })

    user_review = data.get('user_review')
    user_rating = data.get('user_rating')

    if user_review and user_rating:
        if isinstance(user_review, str) and isinstance(user_rating, (int, float)):
            add_review_to_bigquery(user_id, place_id, user_review, user_rating)

    return jsonify({'reviews': result})

if __name__ == '__main__':
    app.run(debug=True)