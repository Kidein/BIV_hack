import pandas as pd
import numpy as np
import torch
import nltk
import pymorphy2
import re
import time

from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import mode

# NTLK
nltk.download('stopwords')
nltk.download('punkt')

morph = pymorphy2.MorphAnalyzer()


def clean_text(text):
    """
    Очистка текста от дат и символов, лемматизация,

    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[№#\-—.,/:\(\)]', ' ', text)
    text = re.sub(r'\bг\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'%', '', text)
    text = re.sub(r'\bт\.ч\.\b', '', text)

    words = word_tokenize(text, language='russian')

    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]

    return ' '.join(lemmatized_words)


# Stopwords
stop_words = set(stopwords.words('russian'))

# Слова, влияющие на эмбеддинги
meaningless_words = {'сумма', 'дог', 'январь', 'февраль', 'март', 'апрель','май', 'июнь', 'июль',
                     'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь', 'ч', 'далее', 'ао', 'оао', 'шт'}


def remove_stopwords(text):
    """"
    Удаление стоп-слов

    """
    words = word_tokenize(text, language='russian')  # Ensure Russian tokenization
    filtered_words = [word for word in words if word.lower() not in stop_words and word.lower() not in meaningless_words]
    return ' '.join(filtered_words)


# Чтение трейна
train = pd.read_csv('/content/payments_training.tsv', sep='\t', header=None)
train = train.set_axis(['index', 'date', 'number', 'text', 'class'], axis=1)
train.drop(columns=(['index', 'date', 'number']), inplace=True)

# Применение функций
train['text'] = train['text'].apply(clean_text)
train['text'] = train['text'].apply(remove_stopwords)


# Кодирование классов
le = preprocessing.LabelEncoder()
le.fit(train['class'])

train['class'] = le.transform(train['class'])
texts, labels = train['text'], train['class']
texts = texts.tolist()


# Чтение теста
test = pd.read_csv('payments_main.tsv', sep='\t', header=None)
test = test[[3]]

test['text'] = test[3].apply(clean_text)
test['text'] = test['text'].apply(remove_stopwords)

test = test['text']
X_test_text = test.tolist()


### Назначение эмбеддинг моделей и параметров XGBoost
models = [
    SentenceTransformer('cointegrated/rubert-tiny2'),
    SentenceTransformer('sergeyzh/rubert-tiny-turbo'),
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
]

xgb_params = [
    {
        'subsample': 0.6,
        'reg_lambda': 3,
        'reg_alpha': 0.1,
        'n_estimators': 250,
        'max_depth': 9,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.9,
        'eval_metric': 'logloss'
    },
    {   'subsample': 0.8,
        'reg_lambda': 3,
        'reg_alpha': 0.1,
        'n_estimators': 250,
        'max_depth': 4,
        'learning_rate': 0.2,
        'gamma': 0,
        'colsample_bytree': 0.8
    },
    {
        'subsample': 0.8,
        'reg_lambda': 2,
        'reg_alpha': 0.1,
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'gamma': 0,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss'
    },
]


def generate_predictions(models, xgb_params, texts, X_test_text, labels):
    """
    Функция использует выбранную ранее конфигурацию моделей и параметров XGBoost.

    Параметры:
        models (list): Список моделей для генерации эмбеддингов.
        xgb_params (list): Список параметров для XGBoost классификатора.
        texts (list): Обучающий набор данных.
        X_test_text (list): Тестовый набор данных.
        labels (list or array-like): Метки классов для обучающего набора данных.

    Returns:
        np.ndarray: Вывод XGBoost, метки классов для X_test_text.
    """
    all_predictions = []

    for model, params in zip(models, xgb_params):
        # Генерация эмбеддингов
        X_train = model.encode(texts)
        X_test = model.encode(X_test_text)

        clf = XGBClassifier(**params)
        clf.fit(X_train, labels)

        y_pred = clf.predict(X_test)
        all_predictions.append(y_pred)

    all_predictions = np.array(all_predictions)
    final_predictions, _ = mode(all_predictions, axis=0)
    return final_predictions.ravel()


def transform_predictions_to_dataframe(predictions, label_encoder, column_name='class'):
    """
    Восстановление истинных меток класса.

    """
    results = label_encoder.inverse_transform(predictions)
    results_df = pd.DataFrame(results, columns=[column_name])
    return results_df



final_predictions = generate_predictions(
    models=models,
    xgb_params=xgb_params,
    texts=texts,
    X_test_text=X_test_text,
    labels=labels
)

results_df = transform_predictions_to_dataframe(final_predictions, le)