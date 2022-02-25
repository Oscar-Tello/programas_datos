#Librerias que se utilizaran en el proyecto
import scattertext as st
import pandas as pd
import spacy

#Libreria que implementa las stopwords
from nltk.corpus import stopwords

def limpieza(csv):
    for row in csv:
        for elemento in row.split():
            




stop_words = set(stopwords.words('spanish')) #Variable con el diccionario en español
diccionarioSimb = {'!','$','=','&','(',')','*',',','-','.','“','”','?','¿','¡','|','°','¬',':','{','}','[',']',
                   '¨','<','>','~','^','♀','♂','!'}

stop_words.update(diccionarioSimb)
nlp = spacy.load('es_core_news_sm')
print(stop_words)

#Variable en la que se guardaran los datos del archivo csv de la ruta especificada
df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\Tweets.csv')

#A ala columna especificada mediante un lamda se limpiaran las stopword del diccionario
df.tweet = df.tweet.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

#Se le aplica una separacion por comas a las palabras de la columna especificada
df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))

#Se quitan palabras menos infrequentes a la o las columnas especificadas
corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=10)

#Se asigna el corpus a la libreria de Scattertext
corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()

#Se asigna a los lados X y Y la etiqueta correspondientes y los valores
dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,
    Y=lambda df: df["Rosengren's S"],
)
dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)

#Creacion de un archivo HTML que mostrara informacion mediante graficas
html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['tweet'],
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'], #Etiquetas de los nombres que se mostraran en la grafica
)

#Accion que guardar en un archivo html con el nombre especificado
open("limpieza_datos.html", 'wb').write(html.encode('utf-8'))
