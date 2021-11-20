import streamlit as st
import nltk
import re
import yaml
import os
import pickle
import pandas as pd
import gc

from pymystem3 import Mystem
from nltk.corpus import stopwords
from string import punctuation

## ----------------------- Initions ----------------------- ##
with open(os.getcwd() + '/app-config.yaml') as f:
    config = yaml.safe_load(f)
    
# for NLP
nltk.download('stopwords')
mystem = Mystem()  # инициализируем для последующего исползования

russian_stopwords = stopwords.words("russian")
eng_stopwords = stopwords.words("english")
symbols = punctuation + '№'
punctuation += '№'
punctuation += '®'

custom_top_words = ['мл','мг','шт','ooo','ру','лп','н','xch','bky','уп','%','п','изм','нд','p','р','n','\n','b','e','е','г','упаковка','пачка','серия','партия','рег','лср',
                    'pul',':','инвойс','rus','вэд','+','rg','g','ltd','d','c','д','кг','-','tyd','дата','оао','ао','зао','ип','инн','лс','код','окпд','россия','ул','область']
numbers = [i for i in range(10)]
numbers = list(map(str, numbers))

with open(config['vectorizer'], 'rb') as f:
    vectorizer = pickle.load(f)
    
codes = pd.read_csv('data/code-info.csv', index_col='code')
code_dict = dict()
for el in codes.iterrows():
    code_dict[el[0]] = el[1][0]
    
st.set_page_config(
        page_title="Кейс №1: ИИ на страже качества российских товаров", layout='wide'
)

## ----------------------- ML ----------------------------- ##
with open(config['model'], 'rb') as f:
    model = pickle.load(f)


## ------------------ PREPROCESSING ----------------------- ##
def preprocess_text(text):
    """
    Очистка и лемматизация текста
    Присутствует специфика данных
    """
    tokens = mystem.lemmatize(text.lower())
    
    text = [re.sub('[0-9][а-я]|\d+', '*', token) for token in tokens if (token.strip() not in russian_stopwords) & \
            (token.strip() not in eng_stopwords) & \
            (token.strip() not in punctuation) & \
            (token.strip() not in numbers) & \
            (token.strip() not in custom_top_words) & \
            (token != ' ')]
    if '*' in text:
        text.remove('*')
        
    text = ' '.join(text)
    
    text = text.replace('*', '')
    text = text.replace('®', '')
    text = text.replace('№', '')
    text = text.replace(')', '')
    text = text.replace('(', '')
    text = text.replace('.', '')
    text = text.replace('"', '')
    text = text.replace('\n', '')
    text = text.replace('«', '')
    text = text.replace('»', '')
    text = text.replace(',,', '')
    text = text.replace(',', '')
    text = text.replace('окпд', '')
    text = text.replace('ooo', '')  # eng
    text = text.replace('ооо', '')  # rus
    text = text.split(' ')
    text = ' '.join([x for x in text if x not in ['']])
    
    return text


## ------------------------ WEB APP ------------------------ ##
st.title('Сервис для определения раздела (кода) Единого перечня продукции')
st.write('В соответствии с постановлением Правительства Российской Федерации от 01.12.2009 № 982 (ЕП РФ)')

code = 0

# input form
with st.form(key='my_form'):
    user_text = st.text_area(label='Текстовое описание продукции', height=15)
    submit_button = st.form_submit_button(label='Определить код продукции')

    if user_text:
        text = preprocess_text(user_text)
        X = vectorizer.transform([text])
        code = model.predict(X)[0]
        
if code != 0:    
    st.markdown(f"> Данному описанию соответствует подкатегория: `{codes[codes.index==code].values[0][0]}`")

uploaded_file = st.file_uploader('Загрузите файл документа', type='xlsx')

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file.getvalue(), engine='openpyxl')
    df = pd.DataFrame()
    
    df['Наименование продукции'] = data[data.columns[1]]
    df['precessed-text'] = data[data.columns[1]].apply(preprocess_text)
    X = vectorizer.transform(df['precessed-text'])
    code = model.predict(X)
    df['Код (определён автоматически)'] = (code/10).round(1)
    df['Код (определён автоматически)'] = df['Код (определён автоматически)'].astype('str')
    df['code'] = code
    df.code = df.code.astype(int)
    df['Подкатегория продукции'] = df['code'].map(code_dict)
    
    if st.button('Сохранить результат'):
        df[['Наименование продукции','Код (определён автоматически)','Подкатегория продукции']].to_csv('Результат анализа кодов по описанию продукции.csv')
        st.markdown(f"""> `Результат определния кодов ЕП сохранён по адресу: 
                    `<span>{os.getcwd()+'/Результат анализа кодов по описанию продукции.csv'}</span>""", unsafe_allow_html=True)
    
    
    st.dataframe(df[['Наименование продукции','Код (определён автоматически)','Подкатегория продукции']])
    
    del data
    del df
    del uploaded_file
    del X
    gc.collect()
