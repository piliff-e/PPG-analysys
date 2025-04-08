# PPG-analysys
## Предсказание поведения пульсовой волны


Вот [Google Dock](https://docs.google.com/document/d/1_jPtP8rYmllPkYXyO8yDc0u4-YSvDAZf6ic81Kd2i0A/edit?tab=t.0) с подробным описанием задачи.


* `/scripts` (сам код):
  - В скрипте `plotting_manually.py` есть комментарии. Там я разбираюсь в том, что в себе содержат скаченные файлы и как грамотно строить по ним графики.
  - Скрипт `plotting_heartpy.py` рисует пульсовую волну буквально в паре строк при помощи библиотеки [HeartPy](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/).


* `/data` (тестовые данные):
  - я взял [отсюда](https://physionet.org/content/wrist/1.0.0/#files-panel).
  - Все ссылки на тестовые данные:
    - https://physionet.org/content/wrist/1.0.0/#files-panel (отсюда брал)
    - https://physionet.org/content/bidmc/1.0.0/#files-panel
    - https://siplab.org/projects/WildPPG
    - https://archive.ics.uci.edu/dataset/495/ppg+dalia