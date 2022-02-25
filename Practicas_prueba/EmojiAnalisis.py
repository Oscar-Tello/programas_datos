import nltk, urllib.request, io, agefromname, zipfile
import scattertext as st
import pandas as pd


with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
    'http://followthehashtag.com/content/uploads/USA-Geolocated-tweets-free-dataset-Followthehashtag.zip'
).read())) as zf:
    df = pd.read_excel(zf.open('dashboard_x_usa_x_filter_nativeretweets.xlsx'))

nlp = st.tweet_tokenzier_factory(nltk.tokenize.TweetTokenizer())
df['parse'] = df['Tweet content'].apply(nlp)

df.iloc[0]