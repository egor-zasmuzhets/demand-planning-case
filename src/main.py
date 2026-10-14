"""Запуск всего пайплайна: python src/main.py

Шаги выполняются последовательно: подготовка данных -> обучение -> прогноз.
Если любой шаг падает, дальнейшие не запускаются.
"""
import time

import data_process
import predict
import train

STEPS = [('data_process', data_process.run), ('train', train.run), ('predict', predict.run)]


def main():
    for name, step in STEPS:
        print(f'\n===== {name} =====')
        t0 = time.time()
        step()
        print(f'----- {name}: {time.time() - t0:.1f} с')


if __name__ == '__main__':
    main()
