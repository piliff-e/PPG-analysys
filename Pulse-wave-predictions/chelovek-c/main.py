import numpy as np
from forecasting import forecast_signal
from replace import analyze_with_heartpy, plot_signals, replace_segments
from shared import load_ppg


def find_bad_segments(original, predicted, threshold=2.0):
    """
    Находит «плохие» сегменты, где ошибка прогноза существенно выше среднего.
    original: numpy array оригинального сигнала
    predicted: numpy array предсказанного сигнала той же (или большей) длины
    threshold: во сколько стандартных отклонений считать аномалией
    Возвращает список пар (start, end) в индексах, где |original - predicted| > threshold * std.
    """
    # Обрежем predicted до длины original, если нужно
    L = min(len(original), len(predicted))
    orig = original[:L]
    pred = predicted[:L]

    error = np.abs(orig - pred)
    std = np.std(error)
    if std == 0:
        std = 1e-6  # защита от деления на ноль
    bad_mask = error > threshold * std

    segments = []
    in_seg = False
    for i, is_bad in enumerate(bad_mask):
        if is_bad and not in_seg:
            start = i
            in_seg = True
        elif not is_bad and in_seg:
            end = i
            segments.append((start, end))
            in_seg = False
    if in_seg:
        segments.append((start, L))
    return segments


def main():
    # Список имён записей (имена WFDB-записей без расширения),
    # в папке data_path должны быть файлы s1_walk.dat/.hea/.atr и т.п.
    filenames = ["s1_walk", "s2_walk", "s3_walk"]
    data_path = "../test_data"  # путь к папке с WFDB-записями
    repeats = 10  # число итераций recursive_forecast

    # Сначала запускаем прогнозирование для всех записей
    # forecast_signal ожидает список имён и data_path
    print("=== Запуск прогнозирования для всех записей ===")
    forecast_results = forecast_signal(filenames, data_path=data_path, repeats=repeats)

    # Для каждой записи: загрузка, анализ прогноза, поиск аномалий и замена
    # for name in filenames:
    name = filenames[0]
    if name in forecast_results:
        print(f"\n--- Обработка записи {name} ---")
        if name not in forecast_results:
            print(f"[!] Прогноз не получен для {name}, пропускаем")
            # continue

        # 1. Загрузка оригинального сигнала
        try:
            # load_ppg из shared: принимает путь к записи без расширения
            signal, fs = load_ppg(f"{data_path}/{name}")
        except Exception as e:
            print(f"[!] Ошибка загрузки сигнала для {name}: {e}")
            # continue

        # 2. Берём прогноз
        forecast_df = forecast_results[name]
        if forecast_df.empty:
            print(f"[!] Прогнозный DataFrame пуст для {name}, пропускаем")
            # continue
        # Извлекаем значения прогноза
        if "NBEATS" not in forecast_df.columns:
            print(f"[!] В прогнозе для {name} нет колонки 'NBEATS', пропускаем")
            # continue
        predicted = forecast_df["NBEATS"].values
        # Обрезаем или дополняем прогноз до длины оригинала:
        if len(predicted) < len(signal):
            # Если прогноз короче, обрезаем оригинал до длины прогноза
            print(
                f"[!] Прогноз короче оригинала ({len(predicted)} < {len(signal)}), обрезаем оригинал"
            )
        # Если прогноз длиннее, можем просто взять первые len(signal) значений
        predicted = predicted[: len(signal)]

        # 3. Поиск «плохих» сегментов
        bad_segments = find_bad_segments(signal, predicted, threshold=2.0)
        print(f"Найдено {len(bad_segments)} плохих сегментов: {bad_segments}")

        # 4. Генерация «хороших» сегментов берём из предсказанного: предсказание считается «хорошим»
        good_segments = []
        for b0, b1 in bad_segments:
            # Если сегмент пустой или выходит за границы, можно пропустить
            if b1 <= b0 or b0 >= len(predicted):
                continue
            # Выбираем срез прогноза как «хороший» сегмент
            seg = predicted[b0:b1]
            good_segments.append(seg)

        # 5. Замена
        try:
            replaced = replace_segments(signal, bad_segments, good_segments)
        except Exception as e:
            print(f"[!] Ошибка при замене сегментов для {name}: {e}")
            # continue

        # 6. Визуализация
        try:
            plot_signals(signal, replaced, fs, bad_segments)
        except Exception as e:
            print(f"[!] Ошибка визуализации для {name}: {e}")

        # 7. HeartPy-анализ
        try:
            analyze_with_heartpy(signal, fs, replaced)
        except Exception as e:
            print(f"[!] Ошибка HeartPy-анализа для {name}: {e}")

    print("\n=== Готово ===")


if __name__ == "__main__":
    main()
