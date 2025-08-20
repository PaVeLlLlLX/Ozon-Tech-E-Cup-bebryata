import re
import pymorphy2
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()
# стоп-слова
russian_stopwords = stopwords.words("russian")
english_stopwords = stopwords.words("english")

def clean_name(text: str) -> str:
    """
    "Легкая" очистка текста для полей вроде бренда или категории.
    1. Приводит к нижнему регистру.
    2. Удаляет все символы, кроме русских, английских букв, цифр и пробелов.
    3. Убирает лишние пробелы.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Оставляем буквы (ru/en), цифры и пробелы
    text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
    # Заменяем несколько пробелов на один
    text = " ".join(text.split())
    
    return text.strip()

def clean_description(text: str) -> str:
    """
    Комплексная функция для очистки текстового описания товара.
    1. Удаляет HTML-теги.
    2. Приводит текст к нижнему регистру.
    3. Удаляет все символы, кроме русских/английских букв, цифр и пробелов.
    4. Токенизирует текст (разбивает на слова).
    5. Удаляет стоп-слова.
    6. Проводит лемматизацию (приводит слова к начальной форме).
    7. Собирает очищенные слова обратно в строку.
    """
    # 0. Проверка на случай, если пришел не текст (например, NaN)
    if not isinstance(text, str):
        return ""

    # 1. Удаляем HTML-теги с помощью BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # 2. Приводим к нижнему регистру
    text = text.lower()

    # 3. Оставляем буквы (ru/en), цифры и пробелы
    text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
    
    # 4. Разбиваем на слова (токенизация) и убираем лишние пробелы
    words = text.split()

    # 5 и 6. Удаляем стоп-слова и лемматизируем
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words if (word not in russian_stopwords and word not in english_stopwords)]

    # 7. Собираем обратно в строку
    cleaned_text = " ".join(lemmatized_words)

    return cleaned_text

# --- Пример использования для проверки ---
if __name__ == '__main__':
    brand_example = "JBL."
    category_example = "Картридж, чернила, тонер"
    name_example = "LeBefane Системный блок (AMD Ryzen 5, RAM 64 ГБ)"
    
    print(f"Бренд '{brand_example}' -> '{clean_name(brand_example)}'")
    print(f"Категория '{category_example}' -> '{clean_name(category_example)}'")
    print(f"Название товара '{name_example}' -> '{clean_description(name_example)}'")