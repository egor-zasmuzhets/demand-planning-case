"""Прогноз на 3 месяца вперёд по рядам сегмента из artifacts/model.json.

Результат: output/forecast.csv (прогноз в единицах источника).
В конце печатает метрики выбранного метода относительно baseline (seasonal naive).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from methods import H, backtest, build_method, metrics_table

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'
OUT_DIR = ROOT / 'output'


def forecast_series(clean, meta, series, model, baseline):
    rows = []
    for s in series:
        g = clean[clean.series == s].sort_values('period')
        y = g.qty_base.to_numpy(float)
        last = pd.Period(g.period.iloc[-1], freq='M')
        # прогноз считаем в нормализованных единицах, в единицы источника возвращаем множителем
        factor = meta.loc[s, 'unit_factor']
        fc = np.clip(model(y, H), 0, None) * factor
        fc_base = np.clip(baseline(y, H), 0, None) * factor
        for k in range(H):
            rows.append({'series': s, 'sku': g.sku.iloc[0], 'location': g.location.iloc[0],
                         'period': str(last + k + 1), 'forecast': round(float(fc[k]), 2),
                         'baseline_forecast': round(float(fc_base[k]), 2)})
    out = pd.DataFrame(rows)
    assert np.isfinite(out.forecast).all() and (out.forecast >= 0).all(), 'некорректный прогноз'
    return out


def print_comparison(title, table, label):
    base, mod = table.loc['baseline'], table.loc[label]
    print(f'\n{title}')
    print(table.rename(index={label: f'model ({label})'}).round(3).to_string())
    print(f'  изменение к baseline: WAPE {100 * (mod.WAPE / base.WAPE - 1):+.1f}%, '
          f'MASE {100 * (mod.MASE / base.MASE - 1):+.1f}%, '
          f'|bias| {100 * abs(base.bias):.1f}% -> {100 * abs(mod.bias):.1f}%')


def run():
    cfg = json.loads((ART_DIR / 'model.json').read_text(encoding='utf-8'))
    clean = pd.read_csv(ART_DIR / 'clean_series.csv')
    meta = pd.read_csv(ART_DIR / 'series_meta.csv', index_col='series')
    model = build_method(cfg['method'], **cfg['params'])
    baseline = build_method(cfg['baseline']['method'], **cfg['baseline']['params'])

    out = forecast_series(clean, meta, cfg['series'], model, baseline)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / 'forecast.csv', index=False)
    print(f'[predict] метод: {cfg["label"]} | рядов: {out.series.nunique()} | '
          f'периоды: {out.period.min()} … {out.period.max()} | сохранено в {OUT_DIR.name}/forecast.csv')
    scaled = meta.loc[cfg['series']].query('unit_factor != 1')
    for s, r in scaled.iterrows():
        print(f'[predict] {s}: прогноз возвращён в единицы источника (x{r.unit_factor:g}), '
              f'ПРОВЕРИТЬ единицы измерения')
    not_fc = meta[~meta.index.isin(cfg['series'])]
    print(f'[predict] не прогнозировались: {len(not_fc)} рядов ({", ".join(sorted(not_fc.segment.unique()))})')

    # метрики относительно baseline: контрольные окна (не участвовали в выборе) и все окна
    bt = backtest(clean, cfg['series'], {'baseline': baseline, cfg['label']: model})
    hold = bt[~bt.origin.isin(cfg['select_origins'])]
    if cfg['label'] == 'seasonal_naive':
        print('\n[metrics] выбран baseline: сравнивать не с чем')
        return
    print_comparison(f'[metrics] КОНТРОЛЬНЫЕ окна ({hold.origin.nunique()} шт., '
                     f'{hold.origin.min()} … {hold.origin.max()}), в выборе параметра не участвовали:',
                     metrics_table(hold), cfg['label'])
    print_comparison(f'[metrics] ВСЕ окна ({bt.origin.nunique()} шт.; первые {len(cfg["select_origins"])} '
                     f'участвовали в выборе, оценка оптимистична):', metrics_table(bt), cfg['label'])


if __name__ == '__main__':
    run()