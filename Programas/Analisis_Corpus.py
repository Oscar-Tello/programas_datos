# Librerias que se utilizaran en el proyecto
import scattertext as st
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import numpy as np
from collections import Counter
import heapq

# Libreria que implementa las stopwords
from nltk.corpus import stopwords


def limpieza(csv):
    for simbolo in diccionarioSimb:
        csv = csv.replace(simbolo, '')
        csv = normalize(csv)
    #print(csv.split())
    csv = ' '.join(word for word in csv.split() if word not in stop_words)
    # for word in stop_words:
    #   word = ' ' + word + ' '
    #  csv = csv.replace(word, ' ')
    return csv

def quitar_frecuentes(csv):
    csv =  ' '.join(word for word in csv.split() if word not in frecuentes)
    return csv

def nube_frecuentes(csv):
    csv = ' '.join(word for word in csv.split() if word in frecuentes)
    return csv

def nube_infrecuentes(csv):
    csv = ' '.join(word for word in csv.split() if word not in frecuentes)
    return csv

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


stop_words = set(stopwords.words('english'))  # Variable con el diccionario en español
diccionarioSimb = {'!', '$', '=', '&', '(', ')', '*', '-', '.', '“', '”', '?', '¿', '¡', '|', '°', '¬', ':', '{', '}',
                   '[', ']', '¨', '<', '>', '~', '^', '♀', '♂', '!', '#', '@', '/', '’', ',','\"'}


# stop_words.update(diccionarioSimb)
nlp = spacy.load('en_core_web_sm')
print(stop_words)

# Variable en la que se guardaran los datos del archivo csv de la ruta especificada

df = pd.read_csv('C:\\Users\\Tello\\Desktop\\programas\\programas_datos\\Datos\\tweets\\UkraineRussia.csv')
#df = pd.read_csv('C:\\Users\\Oscar Tello\\Desktop\\programas_datos\\Datos\\tweets\\WWIII.csv')
#df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\tweets\\STOPPUTTIN.csv')

# limpieza(df['tweet'])

# A ala columna especificada mediante un lamda se limpiaran las stopword del diccionario
df.tweet = df.tweet.apply(lambda x: limpieza(x.lower()))

# Se le aplica una separacion por comas a las palabras de la columna especificada
df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))
#print(df['parse'])

# Se quitan palabras menos infrequentes a la o las columnas especificadas
corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=10)

#print(corpus.get_df()['tweet'])

# Se asigna el corpus a la libreria de Scattertext
corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()

#prueba = corpus.get_df()
#prueba.to_csv("corpus_analisis.csv")

#dispersion_df['Frequency'].nlargest(50).to_csv('Frecuentes.csv')

#print(dispersion_df['Frequency'].nlargest(50))
#print(dispersion_df['Frequency'].nlargest(50).index)

#dispersion_df['Frequency'].index[dispersion_df['Frequency']==73].tolist()
#print(dispersion_df.drop(dispersion_df['Frequency'].nlargest(50).index,axis=0))

frecuentes = dispersion_df['Frequency'].nlargest(30).index
#print(frecuentes)

df = pd.read_csv('C:\\Users\\Tello\\Desktop\\programas\\programas_datos\\Datos\\tweets\\UkraineRussia.csv')
df.tweet = df.tweet.apply(lambda x: limpieza(x.lower()))
nube_palabras1 = df.tweet.apply(lambda x: nube_frecuentes(x.lower()))
nube_palabras2 = df.tweet.apply(lambda x: nube_infrecuentes(x.lower()))

df.tweet = df.tweet.apply(lambda x: quitar_frecuentes(x.lower()))


text = ' '.join(palabra for palabra in nube_palabras1)
#print(text)
word_cloud = WordCloud(collocations= False, background_color= "white").generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

text2 = ' '.join(palabra for palabra in nube_palabras2)
word_cloud2 = WordCloud(collocations=False, background_color= "white").generate(text2)

plt.imshow(word_cloud2, interpolation='bilinear')
plt.axis("off")
plt.show()

df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))

corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=100)

corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()

#dispersion_df = dispersion_df.drop(['nt'],axis=0)
print(dispersion_df['Frequency'].nlargest(50))
print(dispersion_df['Frequency'].nlargest(50).index)

#corpus.get_df().to_csv('analisis_nt.csv')

# Se asigna a los lados X y Y la etiqueta correspondientes y los valores
dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,
    Y=lambda df: df["Rosengren's S"],
)
dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)

# Creacion de un archivo HTML que mostrara informacion mediante graficas
html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['parse'],
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'],
    # Etiquetas de los nombres que se mostraran en la grafica
)

# Accion que guardar en un archivo html con el nombre especificado
open("Dispersion_UkraineRussia.html", 'wb').write(html.encode('utf-8'))
