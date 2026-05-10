import re
import os
import pymorphy3
from transformers import pipeline
from collections import Counter

morph = pymorphy3.MorphAnalyzer()

MODEL_PATH = "./model"

try:
    if os.path.exists(MODEL_PATH):
        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH
        )
    else:
        classifier = None
except:
    classifier = None


bad_triggers = [
    'ужас', 'плохо', 'верните', 'кошмар', 'обман',
    'дорого', 'отстой', 'жаль', 'позор', 'бесит',
    'глупо', 'лохотрон', 'развод', 'мошенники',
    'херня', 'чушь', 'удалите', 'проблема',
    'вранье', 'фу', 'задолбали', 'кидалово'
]

good_triggers = [
    'спасибо', 'супер', 'класс', 'ура', 'лучш',
    'мощно', 'красиво', 'восторг', 'круто',
    'талант', 'умничка', 'огонь', 'браво',
    'обожаю', 'рекомендую', '❤️', '🔥'
]


themes_map = {
    'Деньги': [
        'цена', 'стоимость', 'дорого', 'дешево',
        'деньги', 'оплата', 'купить', 'билет'
    ],
    'Логистика': [
        'где', 'когда', 'адрес', 'доставка',
        'город', 'время', 'маршрут'
    ],
    'Качество': [
        'качество', 'ужас', 'сервис',
        'обман', 'брак', 'проблема'
    ],
    'Поддержка': [
        'ответ', 'поддержка', 'игнор',
        'администратор', 'оператор'
    ]
}


def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9 ]', ' ', text)
    return text.lower()


def get_lemmas(text):
    text = clean_text(text)
    words = text.split()

    lemmas = []

    for word in words:
        try:
            lemmas.append(morph.parse(word)[0].normal_form)
        except:
            pass

    return lemmas


def analyze_sentiment(texts):
    if not classifier:
        return ["Нейтрально"] * len(texts)

    cleaned = [str(t)[:512] for t in texts]

    results = classifier(cleaned)

    final = []

    for i, res in enumerate(results):
        text = cleaned[i].lower()

        if 'label_1' in res['label'].lower():
            pred = 'Позитив'
        elif 'label_2' in res['label'].lower():
            pred = 'Негатив'
        else:
            pred = 'Нейтрально'

        if any(w in text for w in bad_triggers):
            pred = 'Негатив'

        if pred == 'Нейтрально':
            if any(w in text for w in good_triggers):
                pred = 'Позитив'

        final.append(pred)

    return final


def classify_theme(text):
    text = clean_text(text)

    for theme, keywords in themes_map.items():
        if any(k in text for k in keywords):
            return theme

    return "Общее"


def calculate_priority_score(row):
    text = str(row.get('comment_text', '')).lower()

    score = 1.0

    if len(text) > 100:
        score += 1

    if '?' in text:
        score += 1

    if any(w in text for w in bad_triggers):
        score += 2

    return min(round(score, 1), 5.0)


def generate_insights(df):
    insights = []

    negative = df[df['sentiment'] == 'Негатив']

    if not negative.empty:
        top_theme = negative['theme'].mode()[0]
        insights.append(
            f"Основной негатив связан с темой: {top_theme}"
        )

    positive = df[df['sentiment'] == 'Позитив']

    if not positive.empty:
        insights.append(
            f"Позитивных комментариев: {len(positive)}"
        )

    return insights
