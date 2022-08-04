import pandas as pd
import spacy
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
                   '[', ']', '¨', '<', '>', '~', '^', '♀', '♂', '!', '#', '@', '/', '’', ',','\"'+'\''}


# stop_words.update(diccionarioSimb)
nlp = spacy.load('en_core_web_sm')
print(stop_words)

df = pd.read_csv('C:\\Users\\Tello\\Desktop\\programas\\programas_datos\\Datos\\tweets\\WWIII2.csv')

df.tweet = df.tweet.apply(lambda x: limpieza(x.lower()))

df.to_csv('C:\\Users\\Tello\\Desktop\\programas\\programas_datos\\Datos\\tweets\\corpus_limpio.csv')