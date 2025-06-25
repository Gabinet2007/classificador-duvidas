import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# lê o arquivo
df = pd.read_csv('../data/duvidas.csv')

# cria o pipeline com o vetorizador e o modelo naive bayes
# esse modelo classifica os dados lidos de acordo com a frequencia em que as palavras se repetem e a importancia delas
modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())

# treina o modelo
modelo.fit(df['pergunta'], df['categoria'])

# salva em um arquivo o modelo
joblib.dump(modelo, '../modelo_duvidas.pkl')
print("Modelo treinado com sucesso!")
