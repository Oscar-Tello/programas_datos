import pandas as pd
import scattertext as st
import spacy
from IPython.display import IFrame
from IPython.core.display import display, HTML
from pprint import pprint
import seaborn as sns
from scattertext import AssociationCompactor

df = pd.read_csv('C:\\Users\\Oscar Tello\\Desktop\\Datos\\corpus_español2.csv')
df['exito'] = 'inpopular'
df.loc[df['visitas']>5000,['exito']]='popular'
print(df)

print(df.exito.value_counts())
nlp = spacy.load('es_core_news_sm')

df[['letra', 'cancion', 'exito']].head()

corpus = (st.CorpusFromPandas(
    df,
    category_col='exito',
    text_col='letra',
    nlp=st.whitespace_nlp_with_sentences
).build())
corpus.get_term_freq_df().to_csv('term_freqs2.csv')
unigram_corpus = corpus.get_unigram_corpus()

html = st.produce_scattertext_explorer(
    corpus,
    category='popular',
    not_categories=['inpopular'],
    sort_by_dist=False,
    metadata=df['cancion'],
    term_scorer=st.RankDifference(),
)

file_name = 'prueba_scatter.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)
