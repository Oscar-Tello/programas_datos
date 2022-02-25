import scattertext as st
import pandas as pd

df = (pd.read_excel('https://www.wordfrequency.info/files/genres_sample.xls').dropna().set_index('lemma')[['SPOKEN', 'FICTION']].iloc[:1000])
df.head()

term_cat_freq = st.TermCategoryFrequencies(df)

html = st.produce_scattertext_explorer(
    term_cat_freq,
    category='SPOKEN',
    category_name='Spoken',
    not_category_name='Fiction',
)
open('DemoCategoriesFrequencies.html', 'wb').write(html.encode('utf-8'))
