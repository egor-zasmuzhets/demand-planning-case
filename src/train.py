"""Обучение: выбор метода для самых простых рядов (сегмент regular_seasonal).

У метода нет обучаемых весов: «обучение» - это выбор между baseline (seasonal naive) и
кандидатами (seasonal naive с поправкой на рост) по rolling-origin бэктесту.
Выбор делается только на первых N_SELECT_WINDOWS окнах. Остальные окна остаются
контрольными: на них predict.py считает итоговые метрики.

Правило выбора: из кандидатов, у которых |bias| меньше, чем у baseline, берётся
тот, у кого меньше WAPE (и он при этом лучше baseline). Иначе остаётся baseline.

Результат: artifacts/model.json
"""
import json
from pathlib import Path

import pandas as pd

from methods import backtest, build_method, metrics_table

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'

TARGET_SEGMENT = 'regular_seasonal'
N_SELECT_WINDOWS = 5     # первые окна бэктеста - на выбор, остальные - контроль
BASELINE = ('seasonal_naive', {})
CANDIDATES = {
    'growth_phi0.5': ('seasonal_naive_growth', {'phi': 0.5}),
    'growth_phi1.0': ('seasonal_naive_growth', {'phi': 1.0}),
}


def run():
    clean = pd.read_csv(ART_DIR / 'clean_series.csv')
    meta = pd.read_csv(ART_DIR / 'series_meta.csv', index_col='series')
    series = list(meta.index[meta.segment == TARGET_SEGMENT])
    print(f'[train] сегмент {TARGET_SEGMENT}: {len(series)} рядов из {len(meta)} '
          f'(остальные не прогнозируются)')

    specs = {'seasonal_naive': BASELINE, **CANDIDATES}
    methods = {label: build_method(name, **params) for label, (name, params) in specs.items()}
    bt = backtest(clean, series, methods)

    origins = sorted(bt.origin.unique())
    select = origins[:N_SELECT_WINDOWS]
    table = metrics_table(bt[bt.origin.isin(select)])
    print(f'[train] окна выбора: {select[0]} … {select[-1]} ({len(select)} шт.), '
          f'контрольные: {len(origins) - len(select)} шт.')
    print(table.round(3).to_string())

    base = table.loc['seasonal_naive']
    eligible = [m for m in CANDIDATES
                if abs(table.loc[m, 'bias']) < abs(base['bias']) and table.loc[m, 'WAPE'] < base['WAPE']]
    chosen = min(eligible, key=lambda m: table.loc[m, 'WAPE']) if eligible else 'seasonal_naive'
    name, params = specs[chosen]
    print(f'[train] выбрано: {chosen}')

    config = {'segment': TARGET_SEGMENT, 'label': chosen, 'method': name, 'params': params,
              'baseline': {'method': BASELINE[0], 'params': BASELINE[1]},
              'select_origins': select, 'series': series}
    ART_DIR.mkdir(exist_ok=True)
    (ART_DIR / 'model.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[save] {ART_DIR.name}/model.json')


if __name__ == '__main__':
    run()