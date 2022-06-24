# Librerias que se utilizaran en el proyecto
import scattertext as st
import pandas as pd
import spacy

# Libreria que implementa las stopwords
from nltk.corpus import stopwords


def limpieza(csv):
    for simbolo in diccionarioSimb:
        csv = csv.replace(simbolo, '')
        csv = normalize(csv)
    csv = ' '.join(word for word in csv.split() if word not in stop_words)
    # for word in stop_words:
    #   word = ' ' + word + ' '
    #  csv = csv.replace(word, ' ')
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
                   '[', ']', '¨', '<', '>', '~', '^', '♀', '♂', '!', '#', '@', '/', '’',
                   'ukrainerussia', 'ukrainewar', 'russian', 'ukraineunderattack','russiaukrainewar','ukraineinvasion' ,'ukraine', 'russia'}
print(stop_words)

# stop_words.update(diccionarioSimb)
nlp = spacy.load('en_core_web_sm')
# print(stop_words)

# Variable en la que se guardaran los datos del archivo csv de la ruta especificada
df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\tweets\\UkraineRussia.csv')

# limpieza(df['tweet'])

# A ala columna especificada mediante un lamda se limpiaran las stopword del diccionario
df.tweet = df.tweet.apply(lambda x: limpieza(x.lower()))

# Se le aplica una separacion por comas a las palabras de la columna especificada
df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))

# Se quitan palabras menos infrequentes a la o las columnas especificadas
corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=100)

# Se asigna el corpus a la libreria de Scattertext
corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()

print(dispersion_df.Frequency)

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
    metadata=corpus.get_df()['tweet'],
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'],
    # Etiquetas de los nombres que se mostraran en la grafica
)

# Accion que guardar en un archivo html con el nombre especificado
open("limpiezaDatos_UkraineRussia_V2.html", 'wb').write(html.encode('utf-8'))
