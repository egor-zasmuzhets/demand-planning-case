"""Подготовка данных.

Читает data/*.csv, чистит, строит полный календарь, помечает и заполняет «испорченные»
месяцы, считает характеристики рядов и раскладывает их по сегментам.

Результат:
    artifacts/clean_series.csv  - очищенные ряды (qty_base - очищенный спрос)
    artifacts/series_meta.csv   - характеристики и сегмент каждого ряда
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
ART_DIR = ROOT / 'artifacts'

KEY = ['sku', 'location', 'period']
JUMP_RATIO = 20
ADI_INTERMITTENT = 1.32
CV2_LUMPY = 0.49
SEAS_P = 0.01
N_PERM = 500
SEED = 42


# ------------------------------------------------------------ загрузка
def load_raw():
    read = lambda name: pd.read_csv(DATA_DIR / name)
    sales, stock = read('sales_history.csv'), read('stock_history.csv')
    promo, items = read('promo_calendar.csv'), read('item_master.csv')
    print(f'[load] sales={len(sales)}, stock={len(stock)}, promo={len(promo)}, items={len(items)} строк')
    return sales, stock, promo, items


# ------------------------------------------------------------ очистка
def drop_duplicates(sales):
    dup_full = sales.duplicated(keep='first')
    sales = sales[~dup_full]
    conflicts = sales.duplicated(KEY, keep=False)
    if conflicts.any():
        raise ValueError(f'дубли по ключу с разными значениями: {int(conflicts.sum())} строк')
    print(f'[dedup] удалено полных дублей: {int(dup_full.sum())}')
    return sales.reset_index(drop=True)


def merge_stock(sales, stock):
    df = sales.merge(stock, on=KEY, how='left', validate='one_to_one')
    missing = int(df.days_out_of_stock.isna().sum())
    df['days_out_of_stock'] = df.days_out_of_stock.fillna(0)
    print(f'[merge] строк продаж без записи об остатках: {missing}')
    return df


def fix_scale_breaks(df):
    """Ищет в рядах скачок масштаба (смена единиц измерения) и приводит ряд к единой шкале.

    Шкалой считается начало ряда. Множитель округляется до степени десяти.
    Возвращает df и словарь {series: unit_factor}: во сколько раз «текущие» единицы
    источника больше нормализованных (нужен, чтобы вернуть прогноз в единицы источника).
    """
    df = df.sort_values(['sku', 'location', 'period']).reset_index(drop=True)
    df['qty'] = df.qty.astype(float)
    df['stock_end_qty'] = df.stock_end_qty.astype(float)
    factors = {}
    for (sku, loc), g in df.groupby(['sku', 'location']):
        jump = np.log(g.qty.where(g.qty > 0)).diff().abs()
        breaks = jump.index[jump > np.log(JUMP_RATIO)]
        if len(breaks) == 0:
            continue
        if len(breaks) > 1:
            raise ValueError(f'{sku} {loc}: несколько скачков масштаба, нужен ручной разбор')
        i = breaks[0]
        ratio = g.qty[g.index >= i].median() / g.qty[g.index < i].median()
        factor = 10.0 ** round(np.log10(ratio))
        idx = g.index[g.index >= i]
        df.loc[idx, ['qty', 'stock_end_qty']] = df.loc[idx, ['qty', 'stock_end_qty']] / factor
        factors[f'{sku} {loc}'] = factor
        print(f'[scale] {sku} {loc}: скачок с {df.loc[i, "period"]}, поделено на {factor:g} '
              f'(ПРОВЕРИТЬ: единицы измерения)')
    return df, factors


def build_calendar(df, promo):
    """Полный календарь внутри «жизни» каждого ряда + флаги проблемных месяцев."""
    promo_keys = promo[['sku', 'period']].drop_duplicates().assign(flag_promo=True)
    frames = []
    for (sku, loc), g in df.groupby(['sku', 'location']):
        periods = pd.period_range(g.period.min(), g.period.max(), freq='M').astype(str)
        base = pd.DataFrame({'sku': sku, 'location': loc, 'period': list(periods)})
        frames.append(base.merge(g, on=KEY, how='left'))
    cal = pd.concat(frames, ignore_index=True).merge(promo_keys, on=['sku', 'period'], how='left')
    cal['series'] = cal.sku + ' ' + cal.location
    cal['flag_gap'] = cal.qty.isna()
    cal['flag_oos'] = cal.days_out_of_stock.fillna(0) > 0
    cal['flag_neg'] = cal.qty < 0
    cal['flag_promo'] = cal.flag_promo.fillna(False).astype(bool)
    return cal.sort_values(['series', 'period']).reset_index(drop=True)


def impute(cal):
    """Проблемные месяцы -> NaN -> линейная интерполяция внутри ряда (qty_base)."""
    flags = ['flag_gap', 'flag_oos', 'flag_neg', 'flag_promo']
    cal['imputed'] = cal[flags].any(axis=1)
    cal['qty_clean'] = cal.qty.where(~cal.imputed)
    cal['qty_base'] = cal.groupby('series').qty_clean.transform(
        lambda s: s.interpolate(limit_direction='both'))
    print('[impute] заполнено месяцев: ' + ', '.join(f'{f[5:]}={int(cal[f].sum())}' for f in flags)
          + f' | всего ячеек: {int(cal.imputed.sum())} из {len(cal)}')
    return cal


# ------------------------------------------------------------ сегментация
def _season_consistency(res, n_years):
    c = np.corrcoef(res.reshape(n_years, 12))
    return c[np.triu_indices(n_years, 1)].mean()


def build_meta(clean, factors):
    rng = np.random.default_rng(SEED)
    last_period = clean.period.max()
    rows = []
    for s, g in clean.groupby('series'):
        v = g.qty_base.to_numpy(float)
        nz = v[v > 0]
        rows.append({'series': s, 'sku': g.sku.iloc[0], 'location': g.location.iloc[0], 'n': len(v),
                     'start': g.period.iloc[0], 'end': g.period.iloc[-1], 'level': np.median(v),
                     'zero_share': (v == 0).mean(), 'adi': len(v) / max((v > 0).sum(), 1),
                     'cv2': (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan,
                     'vol_12m': v[-12:].sum(), 'unit_factor': factors.get(s, 1.0)})
    meta = pd.DataFrame(rows).set_index('series')


    wide = clean.pivot(index='period', columns='series', values='qty_base')
    meta['derived_from'] = None
    meta['share_of_parent'] = np.nan
    for sku, g in meta.groupby('sku'):
        if len(g) < 2:
            continue
        parent = g.level.idxmax()
        for child in g.index.drop(parent):
            both = wide[[parent, child]].dropna()
            ratio = both[child] / both[parent]
            corr = np.corrcoef(np.log(both[parent]), np.log(both[child]))[0, 1]
            if corr > 0.99 and ratio.std() / ratio.mean() < 0.05:
                meta.loc[child, ['derived_from', 'share_of_parent']] = [parent, ratio.mean()]

    meta['seas'] = np.nan
    meta['seas_p'] = np.nan
    meta['trend_pct_yr'] = np.nan
    cand = meta[(meta.n >= 24) & (meta.n % 12 == 0) & meta.start.str.endswith('-01')
                & (meta.zero_share < 0.2) & (meta.end == last_period) & meta.derived_from.isna()].index
    for s in cand:
        y = np.log(wide[s].dropna().to_numpy())
        t = np.arange(len(y))
        slope = np.polyfit(t, y, 1)
        res = y - np.polyval(slope, t)
        obs = _season_consistency(res, len(y) // 12)
        null = np.array([_season_consistency(rng.permutation(res), len(y) // 12) for _ in range(N_PERM)])
        meta.loc[s, ['seas', 'seas_p', 'trend_pct_yr']] = [obs, (1 + (null >= obs).sum()) / (N_PERM + 1),
                                                           np.expm1(12 * slope[0]) * 100]

    def segment(r):
        if r['end'] < last_period:
            return 'discontinued'
        if pd.notna(r.derived_from):
            return 'derived'
        if r.n < 12:
            return 'new'
        if r.n < 24:
            return 'short'
        if r.adi >= ADI_INTERMITTENT:
            return 'intermittent' if r.cv2 < CV2_LUMPY else 'lumpy'
        return 'regular_seasonal' if r.seas_p < SEAS_P else 'regular_flat'

    meta['segment'] = meta.apply(segment, axis=1)
    print('[segments] ' + ', '.join(f'{k}={v}' for k, v in meta.segment.value_counts().items()))
    return meta


# ------------------------------------------------------------ точка входа
def run():
    sales, stock, promo, items = load_raw()
    sales = drop_duplicates(sales)
    df = merge_stock(sales, stock)
    df, factors = fix_scale_breaks(df)
    cal = impute(build_calendar(df, promo))
    meta = build_meta(cal, factors)

    ART_DIR.mkdir(exist_ok=True)
    cols = ['series', 'sku', 'location', 'period', 'qty', 'qty_base', 'imputed',
            'flag_gap', 'flag_oos', 'flag_neg', 'flag_promo']
    cal[cols].to_csv(ART_DIR / 'clean_series.csv', index=False)
    meta.to_csv(ART_DIR / 'series_meta.csv')
    print(f'[save] {ART_DIR.name}/clean_series.csv ({len(cal)} строк), '
          f'{ART_DIR.name}/series_meta.csv ({len(meta)} рядов)')


if __name__ == '__main__':
    run()