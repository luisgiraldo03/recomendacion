from flask import Flask, escape, request, jsonify

from clasificador import Clasificador

clasificadorI = Clasificador()

app = Flask(__name__)

@app.route("/")
def init():
    return "status ok"

@app.route("/clasificador")
def mostrar():
    return jsonify(clasificadorI.calcular("oldani"))

@app.route("/clasificador1")
def mostrar1():
    return jsonify(clasificadorI.calcular1())

if __name__ == "__main__":
    clasificadorI.leerData()
    clasificadorI.generarMatriz()
    app.run()

