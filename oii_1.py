import random
import re
import nltk

import json

with open("bot.json", "r") as config_file:
    data = json.load(config_file)
INTENTS = data["intents"]

def filter_text(text):
    text.strip()  # удаление лишних пробелов в начале и в конце знаки препинания, поэтому подключаем пакет re
    expression = r'[^\w\s]'  # регулярное выражение = “все что не слово и не пробел»
    # '^'  - это отрицание    \w – это обозначение слов      \s – пробел
    text = re.sub(expression, "", text)  # sub – заменить все “все что не слово и не пробел» на «пустоту» в  text
    return text

def text_match(user_text, example):
    user_text = user_text.lower()  # приводим текст к нижнему регистру .  Для решении проблемы 1
    example = example.lower()
    # Дописать функцию так, что бы все примеры ниже работали

    user_text = filter_text(user_text)  # фильтруем пользовательский ввод
    if user_text.find(example) != -1:
        return True

    text_len = len(user_text)  # длина текста
    difference = nltk.edit_distance(user_text, example) / text_len
    # отношение кол-ва ошибок к длине слова, 1 - слово целиком другое, 0 - слово полностью совподает
    return difference < 0.4
# функция, которая находит намерение пользователя по его тексту  с помощью text_match
def get_intent(user_text):
    for intent in INTENTS:
        examples = INTENTS[intent]["examples"]  # список фраз
        for example in examples:
            if len(filter_text(example)) < 3:
                continue
            if text_match(user_text, example):
                return intent  # найденное намерение подходит к польз. тексту
    return None  # ничего не найдено

def get_random_response(intent):
    return random.choice(INTENTS[intent]["responses"])

X = []
y = []
for intent in INTENTS:
    examples = INTENTS[intent]["examples"]
    for example in examples:
        example = filter_text(example)
        if len(example) < 3:
            continue  # пропускаем текст, если он слишком короткий
        X.append(example)  # добавляем вразу в список х
        y.append(intent)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X)  # Обучаем векторайзер
vecX = vectorizer.transform(X)  # Все тексты преобразуем в вектора

from sklearn.neural_network import MLPClassifier
mlp_classifier  = MLPClassifier(random_state=100)
mlp_classifier.fit(vecX, y)  # Обучение модели
print(mlp_classifier.predict(vectorizer.transform(["здравствуйте"])))

from sklearn.metrics import accuracy_score, f1_score
y_pred = mlp_classifier.predict(vecX)
print("accuracy_score", accuracy_score(y, y_pred)) # Сравниваем y и y_pred
print("f1_score", f1_score(y, y_pred, average="macro")) # Сравниваем y и y_pred
def get_intent_ml(user_text):
    user_text = filter_text(user_text)
    vec_text = vectorizer.transform([user_text])
    intent = model.predict(vec_text)[0]
    # model.predict_proba()
    return intent

def bot(user_text):
    intent = get_intent(user_text)
    if intent:
        return get_random_response(intent)
    intent = get_intent_ml(user_text)
    return get_random_response(intent)


import pandas as pd
import nest_asyncio

nest_asyncio.apply()
TOKEN = "6111624795:AAF11YcpPugykXMEZvtATQTgHBN51WFIGpM"
from telegram import Update  # Обновление пришедшее к нам с серверов ТГ
from telegram.ext import ApplicationBuilder, MessageHandler, filters

# Создаем и настраиваем бот-приложение
app = ApplicationBuilder().token(TOKEN).build()

async def telegram_reply(upd: Update, ctx):
    name = upd.message.from_user.full_name
    user_text = upd.message.text
    print(f"{name}: {user_text}")
    reply = bot(user_text)
    print(f"BOT: {reply}")
    await upd.message.reply_text(reply)


handler = MessageHandler(filters.TEXT, telegram_reply)  # Создаем обработчик текстовых сообщений
app.add_handler(handler)  # Добавляем обработчик в приложение

app.run_polling()
