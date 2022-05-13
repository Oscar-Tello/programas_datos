from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analizar (texto):
    vs = analyzer.polarity_scores(texto)
    #print("{:-<65} {}".format(csv, str(vs.get("neu"))))
    return vs

df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\EN_24-02-2022_STOPWAR.csv')

analyzer = SentimentIntensityAnalyzer()

df = df.assign(
    negativo = df.tweet.apply(lambda x: analizar(x).get("neg")),
    neutro = df.tweet.apply(lambda x: analizar(x).get("neu")),
    positivo =df.tweet.apply(lambda x: analizar(x).get("pos")),
    compound = df.tweet.apply(lambda x: analizar(x).get("compound"))
)

df.to_csv('ejemplo.csv')

print(df)
