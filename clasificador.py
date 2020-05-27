import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
        self.df_ratings = pd.read_csv("ratings.csv", sep=";", header=None)
        self.df_ratings.columns = ["UserId", "DiscoId", "Rating"]
        print(self.df_ratings.head())

    def generarMatriz(self):
        self.n_users = self.df_ratings.UserId.max()  # número de usuarios
        self.n_discos = self.df_ratings.DiscoId.max()  # número de discotecas
        self.ratings = np.zeros((self.n_users, self.n_discos))
        for row in self.df_ratings.itertuples():
            self.ratings[row[1] - 1, row[2] - 1] = row[3]
        sparsity = float(len(self.ratings.nonzero()[0]))
        sparsity /= (self.ratings.shape[0] * self.ratings.shape[1])
        sparsity *= 100

        #creamos conjunto de entrenamiento
        self.ratings_train, self.ratings_test = train_test_split(self.ratings, test_size=0.3, random_state=42)

    def recomendacionDiscotecas(self):
        n_discos = self.ratings_train.shape[1]
        print(n_discos)
        neighbors = NearestNeighbors(n_discos, 'cosine')
        neighbors.fit(self.ratings_train.T)  # quedan en las filas las discos
        top_k_distances, top_k_items = neighbors.kneighbors(self.ratings_train.T, return_distance=True)
        print(top_k_items)#peliculas similares

        #ahora con la distancia del coseno
        k = 5
        neigthbors = NearestNeighbors(k, 'cosine')
        neigthbors.fit(self.ratings_train.T)
        top_k_distances, self.top_k_items = neighbors.kneighbors(self.ratings_train.T, return_distance=True)
        print("------------------------------")
        print("discotecas parecidas a babylon")
        print(type(self.top_k_items))
        print(self.top_k_items[1])  # discotecas mas parecidas a "Babylon"

        self.df_discos = pd.read_csv("discos.csv", sep=";", header=None, encoding='latin1')
        self.df_discos.columns = ["DiscoId", "DiscoName", "Direction", "DiscoPlace", "Type", "Description", "MusicType",
                                  "ExpensiveLevel", "Schedule", "Puntuation", "WebSite", "Tel", "Ubication", "Latitude","Length"]

        records = self.df_discos.iloc[self.top_k_items[1]].to_dict(orient="records")
        # records = json.dumps(records).decode('unicode-escape').encode('utf8')
        records = json.dumps(records, ensure_ascii=False).encode('utf8')
        records_parsed = json.loads(records)
        print(records.decode())
        result = self.df_discos.to_json(orient='records')

        return result

    def recomendarPopulares(self):
        self.df_discos = self.df_discos.sort_values(by="Puntuation", ascending=False).iloc[0:10]
        print(self.df_discos)
        result = self.df_discos.to_json(orient='records')
        print(result)

        return result

    def filtrar(self, music, ubication):
        self.df_discos = pd.read_csv("discos.csv", sep=";", header=None, encoding='latin1')
        self.df_discos.columns = ["DiscoId", "DiscoName", "Direction", "DiscoPlace", "Type", "Description", "MusicType",
                                  "ExpensiveLevel", "Schedule", "Puntuation", "WebSite", "Tel", "Ubication","Latitude","Length"]
        print("------------------------")
        print("Fitro")
        por_ubicacion = self.df_discos['Ubication'] == ubication
        filtradas_ubicacion = self.df_discos[por_ubicacion]

        por_musica = filtradas_ubicacion['MusicType'] == music
        filtrado_musica = filtradas_ubicacion[por_musica]

        result_most_popular = filtrado_musica.sort_values(by="Puntuation", ascending=False).iloc[0:10]
        result = result_most_popular.to_json(orient='records')
        print(filtrado_musica.head())

        return result




























