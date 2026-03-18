"""
Script: 02_eda.py
Description: Exploratory Data Analysis of the processed apartment prices dataset.
             Generates figures saved to reports/figures/.

[GenAI Declaration]
Notebook structure, plotting code, and script conversion were assisted by
Claude Sonnet 4.6 on 2026-03-18.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 110

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH    = PROJECT_ROOT / 'data' / 'processed' / 'master_sales_dataset.csv'
FIG_DIR      = PROJECT_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Data Loading ────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    months_sorted = sorted(df['snapshot_month'].unique())
    df['snapshot_month'] = pd.Categorical(
        df['snapshot_month'], categories=months_sorted, ordered=True
    )
    return df, months_sorted


# ── 2. Missing Values ──────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> None:
    missing = (
        df.isnull().sum()
          .rename('n_missing')
          .to_frame()
          .assign(pct_missing=lambda x: x['n_missing'] / len(df) * 100)
          .sort_values('pct_missing', ascending=False)
    )
    cols_with_missing = missing[missing['n_missing'] > 0].index.tolist()

    print('\n=== Missing Values ===')
    print(missing[missing['n_missing'] > 0].to_string())

    if cols_with_missing:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(
            cols_with_missing,
            missing.loc[cols_with_missing, 'pct_missing'],
            color=sns.color_palette('muted')[0]
        )
        ax.set_xlabel('Missing (%)')
        ax.set_title('Missing Values by Column')
        ax.axvline(5, color='red', linestyle='--', linewidth=0.8, label='5% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'missing_values.png')
        plt.close()


# ── 3. Target Variable: price ──────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame) -> None:
    print('\n=== Price Summary ===')
    print(df['price'].describe().to_string())
    print(f'\nSkewness : {df["price"].skew():.3f}')
    print(f'Kurtosis : {df["price"].kurtosis():.3f}')

    log_price = np.log(df['price'].dropna())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(df['price'].dropna(), bins=100,
                 color=sns.color_palette('muted')[0], edgecolor='none')
    axes[0].set_title('price  (raw)')
    axes[0].set_xlabel('PLN')
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    axes[1].hist(log_price, bins=100,
                 color=sns.color_palette('muted')[1], edgecolor='none')
    axes[1].set_title('log(price)')
    axes[1].set_xlabel('log(PLN)')

    for ax in axes:
        ax.set_ylabel('Count')

    fig.suptitle('Distribution of Apartment Prices', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'price_distribution.png')
    plt.close()

    p01, p99 = df['price'].quantile([0.01, 0.99])
    outliers = df[(df['price'] < p01) | (df['price'] > p99)]
    print(f'\n1st pct : {p01:,.0f} PLN  |  99th pct : {p99:,.0f} PLN')
    print(f'Rows outside [p01, p99] : {len(outliers):,}  '
          f'({len(outliers)/len(df)*100:.2f}%)')


# ── 4. City-level Analysis ─────────────────────────────────────────────────

def plot_city_analysis(df: pd.DataFrame) -> list:
    city_order = (
        df.groupby('city')['price'].median()
          .sort_values(ascending=False).index.tolist()
    )

    city_stats = (
        df.groupby('city')['price']
          .agg(['count', 'median', 'mean', 'std'])
          .rename(columns={'count': 'n', 'median': 'median_price',
                           'mean': 'mean_price', 'std': 'std_price'})
          .reindex(city_order)
    )
    print('\n=== City Statistics ===')
    print(city_stats.round(0).to_string())

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=df, x='city', y='price', order=city_order,
                showfliers=False, palette='muted', ax=ax)
    ax.set_title('Price Distribution by City (outliers hidden)')
    ax.set_xlabel('')
    ax.set_ylabel('Price (PLN)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'price_by_city.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    counts = df['city'].value_counts().reindex(city_order)
    ax.bar(counts.index, counts.values, color=sns.color_palette('muted')[2])
    ax.set_title('Number of Listings per City')
    ax.set_ylabel('Count')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'listings_per_city.png')
    plt.close()

    return city_order


# ── 5. Temporal Analysis ───────────────────────────────────────────────────

def plot_temporal_analysis(df: pd.DataFrame, city_order: list) -> None:
    monthly_counts = df.groupby('snapshot_month', observed=True).size()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly_counts.index.astype(str), monthly_counts.values,
           color=sns.color_palette('muted')[3])
    ax.set_title('Number of Listings per Month')
    ax.set_ylabel('Count')
    ax.set_xlabel('Month')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'listings_per_month.png')
    plt.close()

    monthly_price = df.groupby('snapshot_month', observed=True)['price'].median()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly_price.index.astype(str), monthly_price.values,
            marker='o', linewidth=2)
    ax.set_title('National Median Price over Time')
    ax.set_ylabel('Median Price (PLN)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'national_median_price_over_time.png')
    plt.close()

    city_monthly = (
        df.groupby(['city', 'snapshot_month'], observed=True)['price']
          .median()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    for city in city_order:
        sub = city_monthly[city_monthly['city'] == city]
        ax.plot(sub['snapshot_month'].astype(str), sub['price'],
                marker='o', linewidth=1.5, label=city, markersize=4)
    ax.set_title('Median Price over Time by City')
    ax.set_ylabel('Median Price (PLN)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'city_median_price_over_time.png')
    plt.close()


# ── 6. Continuous Features vs. Price ──────────────────────────────────────

def plot_continuous_features(df: pd.DataFrame) -> None:
    KEY_FEATURES = ['squareMeters', 'centreDistance', 'poiCount', 'rooms']

    df_plot = df[KEY_FEATURES + ['price']].dropna().copy()
    df_plot['log_price'] = np.log(df_plot['price'])
    sample = df_plot.sample(min(10_000, len(df_plot)), random_state=42)

    fig, axes = plt.subplots(1, len(KEY_FEATURES), figsize=(16, 4))
    for ax, feat in zip(axes, KEY_FEATURES):
        ax.scatter(sample[feat], sample['log_price'],
                   alpha=0.15, s=5, color=sns.color_palette('muted')[4])
        ax.set_xlabel(feat)
        ax.set_ylabel('log(price)')
        ax.set_title(feat)
    fig.suptitle('Key Features vs. log(price)  (10k sample)', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'key_features_vs_logprice.png')
    plt.close()

    df['price_per_sqm'] = df['price'] / df['squareMeters']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df['price_per_sqm'].dropna(), bins=100,
                 color=sns.color_palette('muted')[5], edgecolor='none')
    axes[0].set_title('price_per_sqm (raw)')
    axes[0].set_xlabel('PLN / m²')

    axes[1].hist(np.log(df['price_per_sqm'].dropna()), bins=100,
                 color=sns.color_palette('muted')[0], edgecolor='none')
    axes[1].set_title('log(price_per_sqm)')
    axes[1].set_xlabel('log(PLN / m²)')

    for ax in axes:
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'price_per_sqm.png')
    plt.close()

    print('\n=== price_per_sqm Summary ===')
    print(df['price_per_sqm'].describe().to_string())


# ── 7. Categorical Features ────────────────────────────────────────────────

def plot_categorical_features(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ['type', 'ownership']):
        order = df.groupby(col)['price'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y='price', order=order,
                    showfliers=False, palette='muted', ax=ax)
        ax.set_title(f'Price by {col}')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'price_by_type_ownership.png')
    plt.close()

    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator',
                   'hasSecurity', 'hasStorageRoom']
    rows = []
    for col in binary_cols:
        med_yes = df[df[col] == 'yes']['price'].median()
        med_no  = df[df[col] == 'no']['price'].median()
        rows.append({
            'feature': col,
            'median_yes': med_yes,
            'median_no': med_no,
            'uplift_pct': (med_yes - med_no) / med_no * 100
        })
    uplift_df = pd.DataFrame(rows).set_index('feature').round(1)
    print('\n=== Amenity Price Uplift ===')
    print(uplift_df.to_string())


# ── 8. Correlation Matrix ──────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    NUM_COLS = ['price', 'squareMeters', 'rooms', 'floor', 'floorCount',
                'buildYear', 'centreDistance', 'poiCount',
                'schoolDistance', 'clinicDistance', 'restaurantDistance']

    corr = df[NUM_COLS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.4, ax=ax)
    ax.set_title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'correlation_matrix.png')
    plt.close()

    print('\n=== Correlations with price (sorted) ===')
    print(corr['price'].drop('price').sort_values(ascending=False).to_string())


# ── 9. Prior Elicitation Reference ────────────────────────────────────────

def print_prior_reference(df: pd.DataFrame) -> None:
    log_price = np.log(df['price'].dropna())
    log_sqm   = np.log(df['squareMeters'].dropna())

    print('\n=== Prior Elicitation Reference ===')
    print(f'log(price)      mean={log_price.mean():.3f}  std={log_price.std():.3f}')
    print(f'log(sqm)        mean={log_sqm.mean():.3f}    std={log_sqm.std():.3f}')
    print(f'centreDistance  mean={df["centreDistance"].mean():.2f}  '
          f'std={df["centreDistance"].std():.2f}')
    print(f'Number of cities : {df["city"].nunique()}')
    print(f'Number of months : {df["snapshot_month"].nunique()}')
    print('\nPer-city mean log(price):')
    print(
        df.groupby('city', observed=True)
          .apply(lambda g: np.log(g['price']).mean())
          .sort_values(ascending=False)
          .round(3)
          .to_string()
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f'Loading data from: {DATA_PATH}')
    df, months_sorted = load_data(DATA_PATH)
    print(f'Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns')
    print(f'Months : {months_sorted[0]}  →  {months_sorted[-1]}  '
          f'({len(months_sorted)} snapshots)')
    print(f'Cities : {sorted(df["city"].unique())}')

    plot_missing_values(df)
    plot_price_distribution(df)
    city_order = plot_city_analysis(df)
    plot_temporal_analysis(df, city_order)
    plot_continuous_features(df)
    plot_categorical_features(df)
    plot_correlation_matrix(df)
    print_prior_reference(df)

    print(f'\nAll figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
