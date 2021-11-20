import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation


"""
Скрипт по предобработке исходного текста.
На выходе получаем csv с тремя колонками: очищенный и лемматизированный текст из описания продукции, код (target), исходный текст описания продукции
"""


"""
Raw data ---------------------------------------------------- #
"""
data = pd.read_excel('data/raw/Реестр 327 тыс. деклараций ЕП РФ без 140000-200000.xlsx', engine='openpyxl')


"""
stop-words ---------------------------------------------------- #
"""
punctuation += '№'
punctuation += '®'

nltk.download("stopwords")

custom_top_words = ['мл','мг','шт','ooo','ру','лп','н','xch','bky','уп','%','п','изм','нд','p','р','n','\n','b','e','е','г','упаковка','пачка','серия','партия','рег','лср',
                    'pul',':','инвойс','rus','вэд','+','rg','g','ltd','d','c','д','кг','-','tyd','дата','оао','ао','зао','ип','инн','лс','код','окпд','россия','ул','область']
numbers = [i for i in range(10)]
numbers = list(map(str, numbers))

ttd = pd.read_csv('terms_to_delete.txt')
ttd.dropna(inplace=True)
terms_to_delete = ttd.values.flatten().tolist()


"""
Initions ---------------------------------------------------- #
"""
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")
eng_stopwords = stopwords.words("english")


"""
Methods ---------------------------------------------------- #
"""

def return_right_code(codes):
    """
    codes - строка нескольких кодов (тип list)
    """
    arr = np.array(codes.split('; '), dtype=np.float)
    arr2 = np.array(list(map(int, arr)))
    
    if sum(arr2/arr2.min()) == arr.shape:
        """
        если коды одной группы, то берём максимальное значение подгруппы
        """
        code = arr.max()
    else:
        """
        коды разной группы, возвращаем 0
        """
        code = 0

    return code
    
    
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
            # (token.strip() not in terms_to_delete) & \
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
    
    # try:
    #     regex_str = [r"рег. №", "рег №", 'серия', "ру №", "лср-", "лп-", "P N".lower()]
    #     res = re.search("|".join(regex_str), text).start()
    #     text = text[:res]
    # except:
    #     pass
    
    return text


def preprocessing_raw_data(data):
    """
    Предобработка превоначального набора данных
    """
    df = data.copy()
    df.drop(columns=['id','Подкатегория продукции'], inplace=True, axis=1)
    df.rename({'Общее наименование продукции':'description', 'Раздел ЕП РФ (Код из ФГИС ФСА для подкатегории продукции)':'code'}, axis=1, inplace=True)
    
    """
    преобразование кодов:
    - если есть категория и подкатегория - оставляем подкатегорию (так понял эксперта на чек-поинте)
    - если есть несколько категорий/подкатегорий, которые отличаются своей основной категорией - принимаем за махинацию, 
                                                                                                 отбираем в отдельный массиив для ручного реагирования, 
                                                                                                 в обучающую выборку не берём
    """
    df['processed-code'] = df['code'].apply(return_right_code)
    df.drop(index=df[df['processed-code'] == 0].index, inplace=True)
    
    df.code = df['processed-code']
    df.drop('processed-code', axis=1, inplace=True)
    
    
    """
    предобработка текста
    """
    df['clear-description'] = df['description'].apply(preprocess_text)
    
    
    return df[['clear-description','code','description']]


"""
Main ---------------------------------------------------- #
"""
df = preprocessing_raw_data(data)
# удаляем пропуски и дубликаты
df.dropna(inplace=True)
df[['clear-description', 'code']].drop_duplicates(inplace=True)
df.drop(df[df.duplicated()].index, inplace=True)

df.to_csv('data/processed/processed-dataframe.csv', index=False)
