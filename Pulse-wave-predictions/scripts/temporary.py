import numpy as np


def generate_random_bad_segments(
    signal_length, num_segments=5, min_len=100, max_len=500, seed=None
):
    """
    Генерирует список псевдо-плохих сегментов для сигнала длины signal_length.
    Возвращает список кортежей [(start, end), ...].
    Параметры:
      - signal_length: длина сигнала (число отсчётов).
      - num_segments: сколько сегментов сгенерировать.
      - min_len, max_len: минимальная и максимальная длина сегмента (в отсчетах).
      - seed: для воспроизводимости можно передать целое.
    Сегменты могут пересекаться или идти вразнобой; при необходимости можно потом объединять или фильтровать.
    """
    if seed is not None:
        np.random.seed(seed)
    bad_segments = []
    for _ in range(num_segments):
        # случайный старт от 0 до signal_length - min_len - 1
        if signal_length <= min_len:
            break
        start = np.random.randint(0, signal_length - min_len)
        length = np.random.randint(min_len, min(max_len, signal_length - start) + 1)
        end = start + length
        bad_segments.append((start, end))
    # По желанию: отсортировать по start
    bad_segments = sorted(bad_segments, key=lambda x: x[0])
    # Можно дополнительно объединить слишком близкие или пересекающиеся сегменты:
    # Но для простоты оставляем как есть
    return bad_segments
