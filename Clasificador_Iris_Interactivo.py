#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador Iris Interactivo

Created on Wed Oct 27 13:48:12 2021
Modified on Thu Oct 12 13:10 2023


@author: furlan
"""

# Importar librerías

import numpy as np
import pandas as pd

import streamlit as st

from tensorflow.keras.models import load_model
import joblib


modelo_flor = load_model("modelo_final_iris.h5")
normalizador_flor = joblib.load("normalizador_iris.pkl") 

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


def aceptar_datos_usuario():	
    largo_sepalo = st.number_input("Ingrese la longitud del sépalo: ")
    ancho_sepalo = st.number_input("Ingrese el ancho del sépalo: ")
    largo_petalo = st.number_input("Ingrese la longitud del pétalo: ")
    ancho_petalo = st.number_input("Ingrese el ancho del pétalo: ")
    
    datos_flor = {'long_sepalo': largo_sepalo,
                  'ancho_sepalo': ancho_sepalo,
                  'long_petalo': largo_petalo,
                  'ancho_petalo': ancho_petalo}	
    
    return datos_flor


# Funciones básicas para nombrar y describir el dashboard


st.title("Identificación de la clase de flor Iris")
st.sidebar.image('./Datos/iris_flowers.jpeg', width = 300)
'''Esta página es una demo del despliegue de un modelo de una red neuronal
   para predecir la clase de una flor Iris dadas sus dimensiones de sépalo
   y pétalo
'''

# Aceptar los datos del usuario para predecir la especie de Iris	
datos_flor = aceptar_datos_usuario()

# Pasar los datos al modelo para que lo clasifique
    
resultado = devuelve_prediccion(modelo_flor,
                    normalizador_flor,
                    datos_flor)

# Desplegar los datos y el resultado

st.text("Por los siguientes datos: \n")
st.text(f"Longitud del sépalo: {datos_flor['long_sepalo']} cm.")
st.text(f"Ancho del sépalo: {datos_flor['ancho_sepalo']} cm.")
st.text(f"Longitud del pétalo: {datos_flor['long_petalo']} cm.")
st.text(f"Ancho del pétalo: {datos_flor['ancho_petalo']} cm. \n")
st.text(f"La flor muestreada es de la especie {resultado}")
 

