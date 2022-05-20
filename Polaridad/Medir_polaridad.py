from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def medir(texto):
    vs = analyzer.polarity_scores(texto)
    neg = vs.get("neg")
    pos = vs.get("pos")
    neu = vs.get("neu")
    mayor = ""
    if neg >= pos and neg >= neu:
        mayor = "negativo"
    elif neg >= pos and neu >= neg:
        mayor = "neutro"
    elif pos >= neg and pos >= neu:
        mayor = "positivo"
    elif neu >= neg and neu >= pos:
        mayor = "neutro"
    #print(mayor)
    return mayor

df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\EN_24-02-2022_STOPWAR.csv')
analyzer = SentimentIntensityAnalyzer()

#df = df.tweet.apply(lambda x: medir(x))

df = df.assign(
    sentimiento = df.tweet.apply(lambda x: medir(x))
)

df.to_csv("sentimiento.csv")
print(df)
