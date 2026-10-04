"""
Загрузка датасета AI4I 2020 Predictive Maintenance напрямую с Kaggle:
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

Использует официальный REST API Kaggle:
    GET https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}/{file}
Требует авторизации — Kaggle не отдаёт файлы анонимно.

ЧТО НУЖНО ВСТАВИТЬ САМОСТОЯТЕЛЬНО:
  1. KAGGLE_USERNAME и KAGGLE_KEY - получить на https://www.kaggle.com/settings
     -> раздел "API" -> "Create New Token". Скачается файл kaggle.json вида
     {"username": "...", "key": "..."} - оба значения вставить ниже.
  2. Если структура датасета на Kaggle изменится (другой slug/имя файла) -
     поправить DATASET_OWNER / DATASET_SLUG / DATASET_FILE.

Использованы только стандартные модули Python (os, io, zipfile, urllib) -
дополнительных библиотек сверх разрешённого списка не требуется.
"""

import io
import os
import zipfile
import urllib.request

# --- ВСТАВИТЬ СВОИ ЗНАЧЕНИЯ ---
KAGGLE_USERNAME = ""   # TODO: username из kaggle.json
KAGGLE_KEY = ""         # TODO: key из kaggle.json
# -------------------------------

DATASET_OWNER = "stephanmatzka"
DATASET_SLUG = "predictive-maintenance-dataset-ai4i-2020"
DATASET_FILE = "ai4i2020.csv"

API_URL = (f"https://www.kaggle.com/api/v1/datasets/download/"
           f"{DATASET_OWNER}/{DATASET_SLUG}/{DATASET_FILE}")


def download_ai4i_dataset(dest_csv: str = DATASET_FILE,
                           username: str = KAGGLE_USERNAME,
                           key: str = KAGGLE_KEY) -> str:
    """
    Скачивает CSV напрямую с Kaggle (каждый вызов - свежая загрузка,
    существующий локальный файл перезаписывается).

    Возвращает путь к скачанному файлу. При отсутствии учётных данных
    или сетевой ошибке бросает понятное исключение - без молчаливых
    заглушек, чтобы не маскировать проблему.
    """
    if not username or not key:
        raise RuntimeError(
            "Не заданы KAGGLE_USERNAME / KAGGLE_KEY.\n"
            "1. https://www.kaggle.com/settings -> API -> Create New Token\n"
            "2. Впишите значения username и key из kaggle.json в начало "
            "download_data.py (или передайте их аргументами функции)."
        )

    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, API_URL, username, key)
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_mgr)
    )

    with opener.open(urllib.request.Request(API_URL), timeout=60) as response:
        raw = response.read()

    # Kaggle отдаёт либо сырой CSV, либо zip-архив — обрабатываем оба случая
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"В архиве нет .csv: {zf.namelist()}")
            with zf.open(csv_names[0]) as src, open(dest_csv, "wb") as out:
                out.write(src.read())
    else:
        with open(dest_csv, "wb") as out:
            out.write(raw)

    print(f"Скачано напрямую с Kaggle: {dest_csv} "
          f"({os.path.getsize(dest_csv)} байт)")
    return dest_csv


if __name__ == "__main__":
    download_ai4i_dataset()
