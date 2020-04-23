import pandas as pd
import numpy as np
import json
from collections import OrderedDict, defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn

class Clasificador(object):
    __instance = None
    __prueba = 0

    def __new__(cls):
        if Clasificador.__instance is None:
            Clasificador.__instance = object.__new__(cls)
        return Clasificador.__instance

    def pruebas(self):
        self.__prueba += 1
        return self.__prueba

    def leerData(self):
        self.df_users = pd.read_csv("users.csv", sep=';')
        self.df_repos = pd.read_csv("discos.csv", sep=';')
        self.df_ratings = pd.read_csv("ratings.csv", sep=';')

    def generarMatriz(self):
        # ahora crearemos una matriz donde cruzaremos los usuarios con las discos
        self.df_matrix = pd.pivot_table(self.df_ratings, values='rating', index='userId', columns='repoId').fillna(0)
        # calculamos porcentaje de sparcy (porcentaje de 0 que hay que rellenar (predecir))
        ratings = self.df_matrix.values
        self.sparsity = float(len(ratings.nonzero()[0]))
        self.sparsity /= (ratings.shape[0] * ratings.shape[1])
        self.sparsity *= 100
        self.ratings_train, self.ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
        # calculamos en una nueva matriz la similitud entre usuarios
        self.sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings)

    def calcular(self, usuario):
        # "sugeridos para mi"
        # separar las filas y columnas
        self.sim_matrix_train = self.sim_matrix[0:24, 0:24]
        self.sim_matrix_test = self.sim_matrix[24:30, 24:30]
        self.users_predictions = self.sim_matrix_train.dot(self.ratings_train) / np.array([np.abs(self.sim_matrix_train).sum(axis=1)]).T

        data = self.df_users[self.df_users['username'] == usuario]
        self.usuario_ver = data.iloc[0]['userId'] - 1  # resta 1 para obtener el index de pandas
        user0 = self.users_predictions.argsort()[self.usuario_ver]

        tabla = []
        # Veamos los tres recomendados con mayor puntaje en la predic para este usuario
        for i, aRepo in enumerate(user0[-6:]):
            selRepo = self.df_repos[self.df_repos['repoId'] == (aRepo + 1)]
            selRepo['title'].to_dict(OrderedDict)
            td = defaultdict(list)
            selRepoTitle = selRepo['title'].to_dict(td)#aqui esta el error para mañana :D
            tabla.append({"title":selRepoTitle, "puntaje": self.users_predictions[self.usuario_ver][aRepo]})






        return tabla

    def calcular1(self):
        # recomendación usando el vecino más cercano "mas similares"
        k = 8
        neighbors = NearestNeighbors(k, 'cosine')
        neighbors.fit(self.ratings_train)
        top_k_distances, top_k_users = neighbors.kneighbors(self.ratings_train, return_distance=True)
        top_k_distances[self.usuario_ver]
        res = top_k_users[self.usuario_ver]
        users_predicts_k = np.zeros(self.ratings_train.shape)
        for i in range(self.ratings_train.shape[0]):  # para cada usuario del conjunto de entrenamiento
            users_predicts_k[i, :] = top_k_distances[i].T.dot(self.ratings_train[top_k_users][i]) / np.array(
                [np.abs(top_k_distances[i].T).sum(axis=0)]).T
        user0 = users_predicts_k.argsort()[self.usuario_ver][-4:]
        # los tres con mayor puntaje en la predic para este usuario
        for aRepo in user0:
            selRepo = self.df_repos[self.df_repos['repoId'] == (aRepo + 1)]
            print(selRepo['title'], 'puntaje:', users_predicts_k[self.usuario_ver][aRepo])

        return "hola"


















