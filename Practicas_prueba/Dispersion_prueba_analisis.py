import scattertext as st
import spacy
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

def limpieza(csv):
    for simbolo in diccionarioSimb:
        csv = csv.replace(simbolo, '')
        csv = normalize(csv)
    #print(csv.split())
    csv = ' '.join(word for word in csv.split() if word not in palabras_negativas)
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

nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))  # Variable con el diccionario en español
palabras_negativas = {'wasnt', 'mightnt', 'couldnt', 'shant', 'mustnt', 'werent', 'wouldnt', 'arent', 'neednt', 'dont', 'doesnt',
            'isnt', 'wont', 'didnt', 'hadnt', 'havent', 'hasnt', 'shouldnt', 'thatll'}

diccionarioSimb = {'!', '$', '=', '&', '(', ')', '*', '-', '.', '“', '”', '?', '¿', '¡', '|', '°', '¬', ':', '{', '}',
                   '[', ']', '¨', '<', '>', '~', '^', '♀', '♂', '!', '#', '@', '/', ','}

print(stop_words)
#df = st.SampleCorpora.ConventionData2012.get_data().assign(parse=lambda df: df.text.apply(nlp))
#df = st.SampleCorpora.ConventionData2012.get_data()
df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\tweets\\dispersion_WWIII2.csv')
#df.to_json('convention_data_2012.json')

df.tweet = df.tweet.apply(lambda x: limpieza(x.lower()))

df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))

corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=10)

corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()

dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,
    Y=lambda df: df["Rosengren's S"],
)

dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)

html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['date'] + ' (' + corpus.get_df()['time'].str.upper() + ')',
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="S de Rosengren",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'],
)

open("Customizing-Visualization_Frequency_3.html", 'wb').write(html.encode('utf-8'))
