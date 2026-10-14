"""Методы прогноза, бэктест и метрики. Используется в train.py и predict.py.

Контракт метода: f(train: np.ndarray, h: int) -> np.ndarray длины h.
Чтобы добавить новый метод, достаточно написать функцию и добавить её в build_method().
"""
from functools import partial

import numpy as np
import pandas as pd

H = 3
SEASON = 12
MIN_TRAIN = 24


# ---------------------------------------------------------------- методы
def seasonal_naive(train, h=H):
    """Baseline: значение того же месяца год назад."""
    n = len(train)
    return np.array([train[n - SEASON + (i % SEASON)] for i in range(h)], dtype=float)


def seasonal_naive_growth(train, h=H, phi=1.0):
    """Seasonal naive, умноженный на (годовой рост ряда)^phi.

    Годовой рост g = среднее последних 12 мес. / среднее предыдущих 12 мес.
    phi=1 - полная поправка на рост, phi=0.5 - демпфированная, phi=0 - обычный seasonal naive.
    Нужен, потому что baseline занижает прогноз на растущих рядах.
    """
    prev = train[-2 * SEASON:-SEASON].mean()
    last = train[-SEASON:].mean()
    g = last / prev if prev > 0 and last > 0 else 1.0
    return seasonal_naive(train, h) * g ** phi


def build_method(name, **params):
    if name == 'seasonal_naive':
        return seasonal_naive
    if name == 'seasonal_naive_growth':
        return partial(seasonal_naive_growth, **params)
    raise ValueError(f'неизвестный метод: {name}')


# ---------------------------------------------------------------- бэктест
def backtest(clean, series_list, methods, h=H, min_train=MIN_TRAIN):
    """Rolling origin: обучаемся на y[:T], прогнозируем T+1..T+h, сдвигаем T на месяц.

    Заполненные на этапе очистки месяцы (imputed) в оценку не попадают:
    у них нет настоящего факта.
    """
    recs = []
    for s in series_list:
        g = clean[clean.series == s].sort_values('period')
        y = g.qty_base.to_numpy(float)
        imputed = g.imputed.to_numpy(bool)
        per = g.period.to_numpy()
        for T in range(min_train, len(y) - h + 1):
            train = y[:T]
            scale = max(np.abs(train[SEASON:] - train[:-SEASON]).mean(), 1e-9)
            for name, f in methods.items():
                fc = f(train, h)
                for k in range(h):
                    if imputed[T + k]:
                        continue
                    recs.append((s, name, per[T - 1], k + 1, per[T + k], y[T + k], fc[k], scale))
    bt = pd.DataFrame(recs, columns=['series', 'method', 'origin', 'h', 'target', 'actual', 'fc', 'scale'])
    bt['err'] = bt.fc - bt.actual
    bt['abs_err'] = bt.err.abs()
    return bt


# ---------------------------------------------------------------- метрики
def metrics(d):
    """WAPE - по объёму, bias - знаковое смещение (<0 занижение), MASE - равный вес каждого ряда."""
    mase = (d.abs_err / d.scale).groupby(d.series).mean().mean()
    return {'WAPE': d.abs_err.sum() / d.actual.sum(),
            'bias': d.err.sum() / d.actual.sum(),
            'MASE': mase,
            'n': len(d)}


def metrics_table(bt):
    return pd.DataFrame({name: metrics(d) for name, d in bt.groupby('method')}).T