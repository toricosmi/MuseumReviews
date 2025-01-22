import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import dateparser
from aiogram import Bot, Dispatcher, types, Router
from aiogram.types import InputFile
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import Message
from aiogram import F
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import asyncio
from aiogram.types import FSInputFile
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Загрузка модели и токенизатора для предсказания тональности
model = AutoModelForSequenceClassification.from_pretrained("./russian_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./russian_sentiment_model")


def predict_sentiment(review):
    # Токенизация текста
    encoding = tokenizer(review, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        # Получение предсказания модели
        outputs = model(**encoding)
        prediction = outputs.logits.argmax(dim=-1)
    sentiment_mapping = {0: "Негативный", 1: "Позитивный"}
    return sentiment_mapping[prediction.item()]


API_TOKEN = "7598182212:AAFBT1otBi8MufKBmmNjr13Wd6zPnSELjDg"

# Настройка Selenium
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# Создаем бота и диспетчер
bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# Создаем маршрутизатор для обработки сообщений
router = Router()
dp.include_router(router)


def parse_reviews(url):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(15)  # Ждем, пока загрузится страница
    reviews_data = []
    seen_reviews = set()  # Множество для отслеживания уникальных отзывов
    try:
        scroll_container = driver.find_element(By.CLASS_NAME, "scroll__container")
        last_scroll_height = 0
        # Ожидаем появления кнопки сортировки и кликаем на нее
        sort_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@aria-haspopup='true' and @role='button']"))
        )
        sort_button.click()  # Нажимаем на кнопку "По умолчанию"

        time.sleep(2)  # Немного ждем, чтобы меню появилось

        # Ожидаем появления и кликаем на опцию "По новизне" в выпадающем меню
        newest_sort_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@aria-label='По новизне']"))
        )
        newest_sort_option.click()  # Нажимаем на "По новизне"

        time.sleep(5)  # Ждем, пока страница перезагрузится с новыми данными
        while True:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_container)
            new_scroll_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)
            if new_scroll_height == last_scroll_height:
                break
            last_scroll_height = new_scroll_height
            reviews_elements = driver.find_elements(By.CLASS_NAME, "business-review-view")
            for review_element in reviews_elements:
                try:
                    review_text = review_element.find_element(By.CLASS_NAME,
                                                              "business-review-view__body-text").text.strip()
                    # Проверка на дубликат
                    if review_text in seen_reviews:
                        continue  # Пропускаем дубликаты
                    seen_reviews.add(review_text)  # Добавляем новый отзыв в множество

                    stars_element = review_element.find_element(By.CLASS_NAME, "business-rating-badge-view__stars")
                    star_spans = stars_element.find_elements(By.TAG_NAME, "span")
                    gold_stars_count = sum(
                        'business-rating-badge-view__star _full' in star.get_attribute("class") for star in star_spans)
                    review_date = review_element.find_element(By.CLASS_NAME, "business-review-view__date").text.strip()
                except:
                    continue
                reviews_data.append({"review": review_text, "stars": gold_stars_count, "date": review_date})
    finally:
        driver.quit()
    return reviews_data


# Функция для обработки данных и построения графика
def process_data(reviews_data):
    output_file = "data.csv"
    sentiment_counts = {"Позитивный": 0, "Негативный": 0}  # Счётчики тональности

    # Анализ тональности каждого отзыва
    for review in reviews_data:
        review["sentiment"] = predict_sentiment(review["review"])
        sentiment_counts[review["sentiment"]] += 1

    # Сохранение данных в CSV
    with open(output_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Review", "Stars", "Date", "Sentiment"])
        for index, review_data in enumerate(reviews_data, start=1):
            writer.writerow(
                [index, review_data['review'], review_data['stars'], review_data['date'], review_data['sentiment']])

    # Подготовка данных для графика
    df = pd.DataFrame(reviews_data)
    current_year = datetime.now().year
    df['date'] = df['date'].apply(
        lambda x: f"{x} {current_year}" if not any(char.isdigit() for char in x.split()[-1]) else x)
    df['date'] = df['date'].apply(lambda x: dateparser.parse(x))
    df['year'] = df['date'].dt.to_period('Y')
    avg_stars_per_year = df.groupby('year')['stars'].mean()
    count_reviews_per_year = df.groupby('year')['stars'].count()

    # Построение графика
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(avg_stars_per_year.index.astype(str), avg_stars_per_year, color='b', marker='o', label='Средняя оценка')
    ax1.set_xlabel("Год")
    ax1.set_ylabel("Средняя оценка", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.bar(count_reviews_per_year.index.astype(str), count_reviews_per_year, color='gray', alpha=0.5,
            label='Количество отзывов')
    ax2.set_ylabel("Количество отзывов", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    fig.suptitle("Средние оценки и количество отзывов по годам")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig('grafic.png', dpi=300, bbox_inches='tight')

    # Возвращаем результаты
    return output_file, 'grafic.png', df, sentiment_counts


from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


# Функция для создания клавиатуры с кнопками
def create_keyboard():
    # Создаем кнопки
    button_1 = KeyboardButton(text="Анализ музея по ссылке")
    button_2 = KeyboardButton(text="Проверить отзыв на тональность")

    # Создаем клавиатуру
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[button_1], [button_2]],  # Добавляем кнопки в клавиатуру
        resize_keyboard=True,  # Размер клавиатуры
        one_time_keyboard=True,  # Клавиатура исчезает после использования
        input_field_placeholder="Выберите опцию"  # Placeholder в поле ввода
    )

    return keyboard


@router.message(F.text == "/start")
async def send_welcome(message: Message):
    keyboard = create_keyboard()  # Создаем клавиатуру
    await message.reply("Привет! Выберите одну из опций:", reply_markup=keyboard)


from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

# Определяем состояния
class MuseumStates(StatesGroup):
    waiting_for_link = State()

# Обработчик нажатия кнопки "Анализ музея по ссылке"
@router.message(F.text == "Анализ музея по ссылке")
async def handle_url_input(message: Message, state: FSMContext):
    await message.reply("Отправьте мне ссылку на сайт, и я соберу отзывы.")
    await state.set_state(MuseumStates.waiting_for_link)  # Устанавливаем состояние ожидания ссылки

# Обработчик для анализа музея по ссылке
@router.message(MuseumStates.waiting_for_link)
async def handle_museum_link(message: Message, state: FSMContext):
    url = message.text
    if "http" not in url:  # Проверка на корректность ссылки
        await message.reply("Пожалуйста, отправьте корректную ссылку на сайт.")
        return

    await message.reply("Собираю отзывы, это может занять некоторое время...")

    # Здесь вызывается функция парсинга
    reviews_data = parse_reviews(url)
    if not reviews_data:
        await message.reply("Не удалось получить отзывы с указанного сайта.")
    else:
        # Обрабатываем данные и отправляем результаты
        csv_file, graph_file, df, sentiment_counts = process_data(reviews_data)
        await message.reply_document(FSInputFile(csv_file), caption="Отзывы с определением тональности в формате CSV.")
        await message.reply_photo(FSInputFile(graph_file), caption="Динамика изменения средней оценки по годам.")

        # Выводим статистику по тональности
        positive_count = sentiment_counts["Позитивный"]
        negative_count = sentiment_counts["Негативный"]

        # Добавляем в ответ
        await message.reply(
            f"Результат определения тональности отзывов для выбранного музея\n"
            f"Количество положительных отзывов: {positive_count}\n"
            f"Количество отрицательных отзывов: {negative_count}\n\n"
        )

    await state.clear()  # Сбрасываем состояние


# Обработчик для анализа тональности отзыва
@router.message(F.text)
async def handle_sentiment_analysis(message: Message):
    sentiment = predict_sentiment(message.text)
    await message.reply(f"Тональность отзыва: {sentiment}")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
