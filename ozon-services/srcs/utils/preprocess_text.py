import re
import pymorphy2
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

morph = pymorphy2.MorphAnalyzer()
# стоп-слова
try:
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")
except LookupError:
    print("Загрузка стоп-слов NLTK...")
    nltk.download('stopwords')
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")

def clean_brand_commertial4_name(text: str) -> str:
    """
    "Легкая" очистка текста для полей вроде бренда или категории.
    1. Приводит к нижнему регистру.
    2. Убирает лишние пробелы.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Заменяем несколько пробелов на один
    text = " ".join(text.split())
    
    return text.strip()

# TODO: рассмотреть разные варианты и применить извлечение признаков
def clean_name_rus(text: str) -> str:
    """
    "Легкая" очистка текста для полей вроде обозначений и названий.
    1. Приводит к нижнему регистру.
    2. Удаляет разделители.
    3. Убирает лишние пробелы.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Удаляем разделители
    text = re.sub(r'[\/;|]', ',', text)
    # Заменяем несколько пробелов на один
    text = " ".join(text.split())
    
    return text.strip()

def clean_description(text: str) -> str:
    """
    Комплексная функция для очистки текстового описания товара.
    1. Удаляет HTML-теги.
    2. Приводит текст к нижнему регистру.
    3. Удаляет все, что не является буквами, цифрами или стандартными знаками препинания. Оставляет спец. символы, они могут быть важны.
    4. Токенизирует текст (разбивает на слова).
    5. Удаляет стоп-слова.
    6. Проводит лемматизацию (приводит слова к начальной форме).
    7. Собирает очищенные слова обратно в строку.
    """
    # 0. Проверка на случай, если пришел не текст (например, NaN)
    if not isinstance(text, str):
        return ""

    # 1. Используем BeautifulSoup для надежного удаления HTML
    #    separator=' ' гарантирует, что слова не склеятся.
    text = BeautifulSoup(text, "html.parser").get_text(separator=' ')

    # Нижний регистр не убираем
    # 2. Приводим к нижнему регистру
    # text = text.lower()

    # 3. Удаляем все, что не является буквами, цифрами или стандартными знаками препинания.
    #    Оставим спец. символы (точки, дефисы, запятые, ти-ре, плюсы, решетки), они могут быть важны.
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\.\-\,\+/#_]', ' ', text)
    
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
    name_example = "<li>Тип товара Новый</li><li><b>Тип</b> Картридж<li><b>Назначение</b> для лазерных принтеров/МФУ<li><b>Цвет</b> черный<li><b>Цвет картриджа</b> черный<li><b>Модель</b> 071H<li><b>PartNumber/Артикул Производителя</b> 5646C002<li><b>Ресурс, страниц</b> 2500<li><b>Принадлежность к группе</b> Оригинальные<li><b>Принадлежность к подгруппе</b> Тонер-картриджи<li><b>Совместимость</b> i-SENSYS LBP122dw/MF272dw/ MF275dw<li><b>Оригинальность</b> оригинальный<li><b>Повышенная ёмкость/ресурс</b> ДА<li><b>Длина упаковки (ед)</b> 0.365<li><b>Ширина упаковки (ед)</b> 0.11<li><b>Габариты упаковки (ед) ДхШхВ</b> 0.365x0.11x0.135<li><b>Высота упаковки (ед)</b> 0.135<li><b>Вес упаковки (ед)"
    
    print(f"Бренд '{brand_example}' -> '{clean_brand_commertial4_name(brand_example)}'")
    print(f"Категория '{category_example}' -> '{clean_brand_commertial4_name(category_example)}'")
    print(f"Название товара -> '{clean_description(name_example)}'")