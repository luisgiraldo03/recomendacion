from flask import Flask, escape, request, jsonify, json
from flask_cors import CORS

from clasificador import Clasificador

clasificadorI = Clasificador()

app = Flask(__name__)
CORS(app)

@app.route("/")
def init():
    return "status ok"

@app.route("/clasificador", methods=['GET', 'POST'])
def mostrar():
    return clasificadorI.recomendacionDiscotecas()

@app.route("/populares", methods=['GET', 'POST'])
def mostrarPopulares():
    return clasificadorI.recomendarPopulares()

@app.route("/filtrar", methods=['GET', 'POST'])
def mostrarFiltro():
    return clasificadorI.filtrar()

if __name__ == "__main__":
    clasificadorI.leerData()
    clasificadorI.generarMatriz()
    clasificadorI.recomendacionDiscotecas()
    clasificadorI.recomendarPopulares()
    clasificadorI.filtrar()
    app.run()

