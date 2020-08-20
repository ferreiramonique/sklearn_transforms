from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class ChangeGO(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Dados alterados para os desejados:
        data.apply(lambda row: row['NOTA_DE'] if np.isnan(row['NOTA_GO']) else row['NOTA_GO'], axis=1)
        #Retornamos o dataframe:
        return data

       
class Normalizar(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        df_num = data.select_dtypes(include=[np.number])
        df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
        data[df_norm.columns] = df_norm
        return data
