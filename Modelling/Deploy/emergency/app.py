import os
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from google.cloud import bigquery

app = Flask(__name__)

emergency_data = None

credentials_path = "/app/keys/direct-hope-387806-8d2b15781824.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = bigquery.Client()

def load_data(table_name):
    table_ref = client.dataset('Jejaka').table(table_name)
    table = client.get_table(table_ref)
    data = client.list_rows(table).to_dataframe()
    return data

def recommend(lat, lon, k):
    global emergency_data

    lat_lon_input = np.radians(np.array([[lat, lon]]))
    lat_lon_places = np.radians(emergency_data[['lat', 'long']])
    distances = haversine_distances(lat_lon_input, lat_lon_places) * 6371000
    
    emergency_data['distance'] = distances.flatten()
    emergency_data_sorted = emergency_data.sort_values('distance')

    top_k_places = emergency_data_sorted.head(k)
    return top_k_places

emergency_data = load_data("item_data_emergency")

@app.route('/recommend-emergency', methods=['POST'])
def recommend_emergency():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    k = data.get('k', 20)

    if lat is None or lon is None:
        return jsonify({'error': 'Latitude and longitude must be provided.'}), 400

    emergency_places = recommend(lat, lon, k)
    recommended_places_json = emergency_places.to_dict(orient='records')
    return jsonify({'places': recommended_places_json})

if __name__ == '__main__':
    app.run(debug=True)
