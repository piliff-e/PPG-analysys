import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, MSTL, AutoCES
import matplotlib.pyplot as plt


# Ваши данные (предположим, что filtered_ac уже загружен)
# Пример данных:
time = np.arange(1000)  # Временные метки
ppg = np.sin(time * 0.1) + np.random.normal(0, 0.1, 1000)  # Имитация PPG

# Подготовка данных для StatsForecast (формат: уникальный_id, дата, значение)
data = {
    'unique_id': np.ones(len(ppg)),  # Один временной ряд
    'ds': time,  # Временные метки
    'y': ppg  # Значения PPG
}

# Инициализация модели
models = [
    AutoARIMA(season_length=50),  # Автоматический ARIMA (предполагаем частоту пульса ~60 уд/мин)
    MSTL(season_length=[50]),  # Модель с сезонностью
    AutoCES()  # Exponential Smoothing
]

sf = StatsForecast(models=models, freq=1)  # freq=1 (шаг времени = 1 единица)

# Прогноз на 50 шагов вперед
forecast = sf.predict(h=50)
print(forecast)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(time, ppg, label='Исходный сигнал', color='blue')
plt.plot(forecast['ds'], forecast['AutoARIMA'], label='Прогноз (AutoARIMA)', color='red', linestyle='--')
plt.legend()
plt.title('Прогноз PPG с использованием StatsForecast')
plt.show()
