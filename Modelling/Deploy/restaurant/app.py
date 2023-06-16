import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from google.cloud import bigquery
from joblib import load

app = Flask(__name__)

credentials_path = "/app/keys/direct-hope-387806-8d2b15781824.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = bigquery.Client()

def load_data(table_name):
    table_ref = client.dataset('Jejaka').table(table_name)
    table = client.get_table(table_ref)
    data = client.list_rows(table).to_dataframe()
    return data

def add_data_to_bigquery(data, table_name):
    table_ref = client.dataset('Jejaka').table(table_name)
    table = client.get_table(table_ref)
    rows = data.to_dict('records')  
    client.insert_rows(table, rows)

def replace_data_in_bigquery(data, table_name):
    table_ref = client.dataset('Jejaka').table(table_name)
    table = client.get_table(table_ref)

    rows = data.to_dict('records')
    update_commands = []
    for row in rows:
        set_commands = ", ".join([f"{key}={value}" for key, value in row.items() if key != 'user_id'])
        update_command = f"UPDATE `{table.project}.{table.dataset_id}.{table.table_id}` SET {set_commands} WHERE user_id = '{row['user_id']}'"
        update_commands.append(update_command)

    for command in update_commands:
        client.query(command)

def predict_ratings(user_vec, model, item_vecs_filtered, item_data_filtered, top_n=20):
    suser_vec = scalerUser.transform(user_vec)
    sitem_vecs = scalerItem.transform(item_vecs_filtered)

    y_p = model.predict([suser_vec[:, :], sitem_vecs[:, :]])
    y_p_unscaled = scalerTarget.inverse_transform(y_p)

    sorted_index = (-y_p_unscaled.reshape(-1)).argsort()
    sorted_ypu = y_p_unscaled[sorted_index]

    top_n_indices = sorted_index[:top_n]
    top_n_ratings = sorted_ypu[:top_n]

    selected_columns = ['place_id', 'place_name', 'place_address', 'image', 'desc', 'total_review', 'ave_rating']
    top_n_predictions = item_data_filtered.iloc[top_n_indices][selected_columns].copy()
    top_n_predictions['Predicted_Rating'] = top_n_ratings
    return top_n_predictions

def is_user_exist(user_id):
    query = f"SELECT user_id FROM Jejaka.user_data_restaurant WHERE user_id = '{user_id}'"
    query_job = client.query(query)
    results = query_job.result()
    return len(list(results)) > 0

scalerTarget_path = "/app/scalerTarget.joblib"
scalerUser_path = "/app/scalerUser.joblib"
scalerItem_path = "/app/scalerItem.joblib"

scalerTarget = load(scalerTarget_path)
scalerUser = load(scalerUser_path)
scalerItem = load(scalerItem_path)

item_data = load_data("item_data-restaurant")
item_vecs = item_data.iloc[:, 7:].values

model_path = os.environ.get('MODEL_PATH')
model = tf.keras.models.load_model(model_path)

@app.route('/recommend-restaurant', methods=["POST"])
def recommend():
    data = request.get_json()

    user_id = data.get('user_id', '0')
    place_id = data.get('place_id', '')
    city = data.get('city/regency', '')

    new_items = {key: 4 * data.get(f'new_{key}', 0) for key in [
        "art_gallery", "bakery", "bar", "cafe", "food", "liquor_store", "lodging", "meal_delivery", "meal_takeaway", "night_club",
        "restaurant", "school", "store"]}

    if is_user_exist(user_id):
        user_data = load_data("user_data_restaurant")
        item_vec = item_data[item_data["place_id"] == place_id].to_numpy()[:, 8:]
        user_vec = user_data[user_data['user_id'] == user_id].to_numpy()
        if len(item_vec) > 0:
            user_vec[:, 1:] = (item_vec * 4 + user_vec[:, 1:]) / 2
            user_data.loc[user_data["user_id"] == user_id, user_data.columns[1:]] = user_vec[:, 1:]
            replace_data_in_bigquery(user_data, "user_data_restaurant")
    else:
        user_vec = [[user_id] + list(new_items.values())]
        user_data = pd.DataFrame(user_vec, columns=['user_id'] + list(new_items.keys()))
        add_data_to_bigquery(user_data, "user_data_restaurant")

    if city:
        item_data_filtered = item_data[item_data['city_regency'] == city]
        item_vecs_filtered = item_data_filtered.iloc[:, 7:].values
    else:
        item_data_filtered = item_data
        item_vecs_filtered = item_vecs

    user_vec = np.array(user_vec)
    user_vecs = np.repeat(user_vec[:, 1:], len(item_vecs_filtered), axis=0)

    predictions = predict_ratings(user_vecs, model, item_vecs_filtered, item_data_filtered, top_n=20)
    predictions_json = predictions.to_dict(orient='records')
    return jsonify({'places': predictions_json})

if __name__ == '__main__':
    app.run(debug=True)
