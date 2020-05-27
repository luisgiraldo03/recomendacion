from flask import Flask, escape, request, jsonify, json
from flask_request_params import bind_request_params
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
    music = request.args['music']
    numberPeople = request.args.get('numberPeople')
    ubication = request.args['ubication']
    preference = request.args.get('preference')
    return clasificadorI.filtrar(music, ubication)

if __name__ == "__main__":
    clasificadorI.leerData()
    clasificadorI.generarMatriz()
    clasificadorI.recomendacionDiscotecas()
    clasificadorI.recomendarPopulares()
    #clasificadorI.filtrar("", "", "", "", "")
    app.run()

