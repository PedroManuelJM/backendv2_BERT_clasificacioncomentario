from flask import Flask, request, jsonify
from transformers import pipeline
from huggingface_hub import login
import pymysql
from flask_cors import CORS
from dotenv import load_dotenv
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

# Cargar variables del archivo .env
load_dotenv()

TOKEN = os.getenv('TOKEN')
# Autenticar con Hugging Face
login(token=TOKEN)  # Reemplaza 'tu_token' con tu token de Hugging Face

# Crear pipeline de clasificación usando Hugging Face
classifier = pipeline("text-classification", model="pedrojm/modelv2_clasificacioncomentario")

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)

# Etiquetas de las clases y puntajes
CLASSES = ["Positivo", "Negativo", "Neutro", "Invalido"]
PUNTAJES = {"Positivo": 5, "Negativo": 1, "Neutro": 3, "Invalido": 0}

@app.route('/')
def home():
    return "Servidor funcionando", 200

# Ruta para clasificar un comentario
@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        user_id = data.get('user_id')
        user_comment = data.get('user_comment')

        if not product_id or not user_id or not user_comment:
            return jsonify({"error": "Faltan datos necesarios en la solicitud"}), 400

        # Generar la fecha actual
        date_comment = datetime.now().strftime('%Y-%m-%d')

        # Clasificar usando el pipeline
        classification_result = classifier(user_comment)
        predicted_class = classification_result[0]["label"]
        clase = predicted_class
        puntaje = PUNTAJES.get(clase, 0)

        # Retornar los resultados
        resultados = [{
            "product_id": product_id,
            "user_comment": user_comment,
            "date_comment": date_comment,
            "classification": clase,
            "rating": puntaje
        }]
        return jsonify(resultados), 200

    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        return jsonify({"error": "Hubo un problema procesando la solicitud"}), 500

if __name__ == '__main__':
    app.run(port=5000)
