from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def medir(texto):
    vs = analyzer.polarity_scores(texto)
    neg = vs.get("neg")
    pos = vs.get("pos")
    neu = vs.get("neu")

    print("negativo: " + str(neg) + ", positivo: " + str(pos) + ", neutro: " + str(neu))


df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\EN_24-02-2022_STOPWAR.csv')
analyzer = SentimentIntensityAnalyzer()

df = df.tweet.apply(lambda x: medir(x))

# df = df.assign(
# negativo = df.tweet.apply(lambda x: medir(x)))
