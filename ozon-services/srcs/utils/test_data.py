import pandas as pd
import os # os нужен для примера, в самой функции не используется

def get_ids_with_empty_column(df: pd.DataFrame, 
                              column_to_check: str, 
                              item_id_col: str = 'ItemID') -> list:
    """
    Находит и возвращает список ID тех строк, у которых указанная колонка
    содержит пустое значение (NaN, None, '', ' ').

    Args:
        df (pd.DataFrame): Исходный DataFrame для поиска.
        column_to_check (str): Название колонки, которую нужно проверить на пустоту.
        item_id_col (str): Название колонки с ID элемента (по умолчанию 'ItemID').

    Returns:
        list: Список ID, для которых значение в column_to_check является пустым.
    """
    print(f"Поиск ID для пустых значений в колонке '{column_to_check}'...")

    # 1. Проверяем, существуют ли необходимые колонки
    if item_id_col not in df.columns:
        print(f"⚠️  Ошибка: Колонка ID '{item_id_col}' не найдена в DataFrame.")
        return [] # Возвращаем пустой список
    if column_to_check not in df.columns:
        print(f"⚠️  Ошибка: Проверяемая колонка '{column_to_check}' не найдена в DataFrame.")
        return []

    # 2. Создаем boolean-маску для поиска всех вариантов "пустых" значений.
    # .fillna('') - заменяет NaN/None на пустую строку.
    # .astype(str) - на всякий случай преобразует все в строки.
    # .str.strip() - убирает пробелы в начале и в конце.
    # == '' - проверяет, осталась ли в итоге пустая строка.
    # Этот метод надежно ловит все 4 типа пустых значений.
    is_empty_mask_1 = df[column_to_check].fillna('').astype(str).str.strip() == ''
    is_empty_mask_2 = df[column_to_check].fillna('').astype(str).str.strip() == '__MISSING__'
    
    # 3. Используем маску, чтобы выбрать ID из нужных строк, и преобразуем в список
    empty_ids_list_1 = df.loc[is_empty_mask_1, item_id_col].tolist()
    empty_ids_list_2 = df.loc[is_empty_mask_2, item_id_col].tolist()
    empty_ids_list = empty_ids_list_1 + empty_ids_list_2
    
    found_count = len(empty_ids_list)
    
    print("Поиск завершен.")
    if found_count > 0:
        print(f"  Найдено {found_count} строк с пустыми значениями.")
        # Выведем несколько примеров для наглядности
        print(f"  Пример найденных ID: {empty_ids_list[:5]}")
    else:
        print(f"  Строк с пустыми значениями в колонке '{column_to_check}' не найдено.")
    
    return empty_ids_list

def display_items_by_id(df: pd.DataFrame, 
                        item_id_list: list, 
                        item_id_col: str = 'ItemID', 
                        description_col: str = 'description') -> pd.DataFrame:
    """
    Находит в DataFrame строки по списку ID и выводит в консоль их 'ItemID' и 'description'.

    Args:
        df (pd.DataFrame): Исходный DataFrame для поиска.
        item_id_list (list): Список ID элементов, которые нужно найти и вывести.
        item_id_col (str): Название колонки с ID элемента (по умолчанию 'ItemID').
        description_col (str): Название колонки с описанием (по умолчанию 'description').

    Returns:
        pd.DataFrame: DataFrame, содержащий только найденные строки.
    """
    print(f"Поиск {len(item_id_list)} элементов в DataFrame...")

    # 1. Проверяем, что список ID не пустой
    if not item_id_list:
        print("Список ID для поиска пуст. Вывод невозможен.")
        return pd.DataFrame() # Возвращаем пустой DataFrame

    # 2. Убеждаемся, что все необходимые колонки существуют в DataFrame
    required_cols = [item_id_col, description_col]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️  Ошибка: Одна из колонок {required_cols} не найдена в DataFrame.")
        return pd.DataFrame()
        
    # 3. Фильтруем DataFrame, чтобы найти строки с нужными ID.
    # Приводим оба списка к строкам для корректного сравнения
    ids_to_find = {str(item_id) for item_id in item_id_list}
    df_found = df[df[item_id_col].astype(str).isin(ids_to_find)].copy()
    
    # 4. Проверяем, найден ли хоть один элемент
    if df_found.empty:
        print("Ни один из указанных ItemID не найден в DataFrame.")
        return df_found

    # 5. Выводим информацию по каждому найденному элементу
    print("\n--- Найденные элементы ---")
    for index, row in df_found.iterrows():
        print(f"ItemID: {row[item_id_col]}")
        # Выводим только первые 200 символов описания, чтобы не засорять консоль
        description_preview = str(row[description_col])[:200]
        if len(str(row[description_col])) > 200:
            description_preview += "..."
        print(f"Description: {description_preview}")
        print("-" * 25) # Разделитель для читаемости
    
    print("\nПоиск завершен.")
    print(f"  Найдено {len(df_found)} из {len(item_id_list)} запрошенных элементов.")
    
    return df_found

# --- Пример использования ---
if __name__ == '__main__':
    key = 'train'
    result_data_path = './data/'
    result_data_path += 'processed/'
    df = pd.read_csv(os.path.join(result_data_path, f'{key}.csv'))
    ids_to_display = get_ids_with_empty_column(df, 'CommercialTypeName4')
    # 'description', 'name_rus', 'brand_name', 'CommercialTypeName4'

    raw_data_path = './data/'
    raw_data_path += f'{key}/'
    df = pd.read_csv(os.path.join(raw_data_path, f'ml_ozon_сounterfeit_{key}.csv'))
    # Список ItemID, которые мы хотим найти
    # Вызываем нашу новую функцию
    found_items_df = display_items_by_id(df, item_id_list=ids_to_display)

    print("\nDataFrame, который вернула функция:")
    print(found_items_df)