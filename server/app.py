import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS  # Importa CORS

house_rent = joblib.load('house_rent_model.pkl')

def predict_house_rent(model, dataframe):
    predictions = model.predict(dataframe)
    return predictions[0]

app = Flask(__name__)
CORS(app)  # Habilita CORS para tu aplicación Flask

@app.route('/modelo', methods=['POST'])
def getProducts():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'El cuerpo de la solicitud debe contener datos JSON'}), 400

        input_data = {
            'city': [data.get('city', 1)],
            'area': [data.get('area', 70)],
            'rooms': [data.get('rooms', 2)],
            'bathroom': [data.get('bathroom', 1)],
            'furniture': [data.get('furniture', 1)],
            'fire insurance (R$)': [data.get('fire insurance (R$)', 42)]
        }

        input_df = pd.DataFrame(input_data)
        pred = predict_house_rent(house_rent, input_df)
        return jsonify({'prediccion': pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
