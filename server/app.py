from tensorflow.keras.models import load_model
import joblib
import numpy as np
from flask import Flask, jsonify, request

modelo_flor = load_model("modelo_final_iris.h5")
normalizador_flor = joblib.load("normalizador_iris.pkl")

muestra_flor = {'long_sepalo':5.1,
                 'ancho_sepalo':3.5,
                 'long_petalo':1.4,
                 'ancho_petalo':0.2}

def devuelve_prediccion(modelo, normalizador, muestra_json):
    
    # Si fueran muchas más características,
    #   probablemente sería bueno codificar
    #   una iteración que contruya este arreglo
    
    long_sep = muestra_json['long_sepalo']
    ancho_sep = muestra_json['ancho_sepalo']
    long_pet = muestra_json['long_petalo']
    ancho_pet = muestra_json['ancho_petalo']
    
    flor = [[long_sep, ancho_sep,
             long_pet, ancho_pet]]
    
    flor = normalizador.transform(flor)
    
    clases = np.array(['Iris-setosa', 
                       'Iris-versicolor', 
                       'Iris-virginica'])
    
    #clase_ind = modelo.predict_classes(flor)
    clase_ind = np.argmax(modelo.predict(flor), axis = -1)    
    
    return clases[clase_ind][0]     


# pred = devuelve_prediccion(modelo_flor,normalizador_flor,muestra_flor)

# print("Esta es la prediccion con el modelo que se es: "+str(pred))


app = Flask(__name__)
@app.route('/modelo', methods=['GET'])
def getProducts():
    pred = devuelve_prediccion(modelo_flor,normalizador_flor,muestra_flor)
    # data = request.get_json()
    # print(data)
    return jsonify({'prediccion': pred})

if __name__ == '__main__':
    app.run(debug=True, port=5000)