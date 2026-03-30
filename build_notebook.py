#!/usr/bin/env python3
"""
build_notebook.py
Programmatically builds a Jupyter notebook (.ipynb) for statistical analysis
of a 2011 BMW 328i xDrive (Schererville, IN listing).

Generates: C:\Users\lcawley\bridge\analysis.ipynb
"""

import json
import os

# ─── Notebook scaffolding helpers ──────────────────────────────────────────────

def make_nb():
    """Return an empty notebook dict (nbformat 4.5)."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.3"
            }
        },
        "cells": []
    }

_cell_id = 0
def _next_id():
    global _cell_id
    _cell_id += 1
    return f"cell-{_cell_id:04d}"

def md(source: str):
    """Return a markdown cell dict."""
    return {
        "id": _next_id(),
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(True)
    }

def code(source: str):
    """Return a code cell dict."""
    return {
        "id": _next_id(),
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().splitlines(True),
        "outputs": [],
        "execution_count": None
    }

# ─── Build cells ───────────────────────────────────────────────────────────────

cells = []

# ============================================================================
# Cell 0 – Title / Intro (markdown)
# ============================================================================
cells.append(md(r"""
# 2011 BMW 328i xDrive (Schererville, IN) — Statistical Background Check

**Purpose:** Place one specific used car within the *population distribution*
of all 2011 BMW 328i vehicles using real market listings and NHTSA safety data.

| Attribute | Value |
|-----------|-------|
| **VIN** | WBAPK7C51BA820431 |
| **Engine** | N52 3.0 L Inline-6, naturally aspirated |
| **Drivetrain** | xDrive AWD |
| **Odometer** | 147,933 mi |
| **Asking price** | $4,800 |
| **History** | 0 accidents · 3 owners · near-100 % BMW dealer service |
| **Known work** | Valve cover @ 86 k, PS pump @ 133 k |
| **Unknown status** | Water pump, oil-pan gasket, front struts |

### Data sources

| File | Records | Description |
|------|---------|-------------|
| `market_listings_raw.json` | 15 listings | Current AutoList.com listings for 2011 BMW 328i (scraped 2025-03-29) |
| `complaints_2011_328I.json` | 678 complaints | All NHTSA complaints for 2011 BMW 328I |
| `complaints_with_mileage.json` | 210 complaints | Subset with mileage extracted from narrative |
| `complaints_2012_328I.json` | 143 complaints | 2012 model-year (comparison cohort) |
| `recalls_2011_328I.json` | 7 recalls | All NHTSA recall campaigns for 2011 BMW 328I |
"""))

# ============================================================================
# Cell 1 – Imports & data loading (code)
# ============================================================================
cells.append(code(r"""
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import json, os, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import gaussian_kde

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.size': 10,
    'legend.fontsize': 9,
})

DATA = Path(r'C:\Users\lcawley\bridge\data')

# ── Load JSON helpers ──────────────────────────────────────────────────────
def load_json(name):
    with open(DATA / name, encoding='utf-8') as f:
        return json.load(f)

market_raw    = load_json('market_listings_raw.json')
complaints_11 = load_json('complaints_2011_328I.json')
complaints_12 = load_json('complaints_2012_328I.json')
mileage_data  = load_json('complaints_with_mileage.json')
recalls_raw   = load_json('recalls_2011_328I.json')

# Convenience
listings = market_raw['listings']
c11      = complaints_11['results']
c12      = complaints_12['results']
recalls  = recalls_raw['results']

# Subject car constants
OUR_PRICE   = 4800
OUR_MILEAGE = 147_933

print(f"{'Dataset':<35} {'Records':>7}")
print("-" * 44)
print(f"{'Market listings':<35} {len(listings):>7}")
print(f"{'2011 NHTSA complaints':<35} {len(c11):>7}")
print(f"{'2012 NHTSA complaints (comparison)':<35} {len(c12):>7}")
print(f"{'Complaints with mileage':<35} {len(mileage_data):>7}")
print(f"{'NHTSA recalls':<35} {len(recalls):>7}")
"""))

# ============================================================================
# Cell 2 – Markdown: Price Distribution
# ============================================================================
cells.append(md(r"""
## 1. Market Position: Price Distribution

We compare the $4,800 asking price against 15 current AutoList.com listings for the
same model year (2011 BMW 328i, all trims). The left panel shows the probability
density (histogram + KDE), and the right panel shows the cumulative distribution
function (CDF). A vertical red line marks our car.
"""))

# ============================================================================
# Cell 3 – Price distribution plot (code)
# ============================================================================
cells.append(code(r"""
prices = np.array([l['price'] for l in listings])

# Empirical percentile (% of listings with price <= ours)
pct_price = (prices <= OUR_PRICE).sum() / len(prices) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Histogram + KDE ──────────────────────────────────────────────────
bins = np.linspace(prices.min() - 500, prices.max() + 500, 16)
ax1.hist(prices, bins=bins, density=True, alpha=0.45, color='steelblue',
         edgecolor='white', label='Listings (n=15)')
kde = gaussian_kde(prices, bw_method=0.35)
xs = np.linspace(prices.min() - 1000, prices.max() + 1000, 300)
ax1.plot(xs, kde(xs), color='navy', lw=2, label='KDE')
ax1.axvline(OUR_PRICE, color='red', ls='--', lw=2, label=f'Our car ${OUR_PRICE:,}')
ax1.annotate(f'${OUR_PRICE:,}\n({pct_price:.0f}th %-ile)',
             xy=(OUR_PRICE, kde([OUR_PRICE])[0]),
             xytext=(OUR_PRICE + 1200, kde([OUR_PRICE])[0] + 0.00005),
             fontsize=10, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
ax1.set_xlabel('Listing Price ($)')
ax1.set_ylabel('Density')
ax1.set_title('Price Distribution of 2011 BMW 328i Listings')
ax1.legend()
ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

# ── Right: CDF ─────────────────────────────────────────────────────────────
sorted_p = np.sort(prices)
cdf_y = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
ax2.step(sorted_p, cdf_y, where='post', color='steelblue', lw=2, label='Empirical CDF')
ax2.axvline(OUR_PRICE, color='red', ls='--', lw=2)
# Horizontal line to CDF curve
interp_y = np.interp(OUR_PRICE, sorted_p, cdf_y)
ax2.hlines(interp_y, ax2.get_xlim()[0] if sorted_p.min() > OUR_PRICE else sorted_p.min() - 500,
           OUR_PRICE, color='red', ls=':', lw=1.5)
ax2.plot(OUR_PRICE, interp_y, 'ro', ms=8, zorder=5)
ax2.annotate(f'{pct_price:.0f}th %-ile',
             xy=(OUR_PRICE, interp_y),
             xytext=(OUR_PRICE + 1500, interp_y - 0.12),
             fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
ax2.set_xlabel('Listing Price ($)')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('CDF of Listing Prices')
ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax2.legend()

plt.tight_layout()
plt.savefig(DATA / 'fig1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*55}")
print(f"  PRICE ANALYSIS  (n = {len(prices)} listings)")
print(f"{'='*55}")
print(f"  Market range : ${prices.min():,.0f} – ${prices.max():,.0f}")
print(f"  Market mean  : ${prices.mean():,.0f}")
print(f"  Market median: ${np.median(prices):,.0f}")
print(f"  Our price    : ${OUR_PRICE:,}")
print(f"  Percentile   : {pct_price:.1f}th  (only {(prices <= OUR_PRICE).sum()} of {len(prices)} listings ≤ ${OUR_PRICE:,})")
print(f"{'='*55}")
"""))

# ============================================================================
# Cell 4 – Markdown: Mileage Distribution
# ============================================================================
cells.append(md(r"""
## 2. Mileage Distribution: Where 148k Sits

Our car's odometer reads **147,933 miles**, which is **above the maximum mileage**
among the 15 current AutoList listings (max ≈ 140 k). This means the car sits
beyond the right tail of what's currently offered on the open market — most
328i's at this mileage have already been bought, scrapped, or de-listed because
the price floor makes listing them uneconomical.
"""))

# ============================================================================
# Cell 5 – Mileage distribution plot (code)
# ============================================================================
cells.append(code(r"""
miles = np.array([l['mileage'] for l in listings])

pct_mileage = (miles <= OUR_MILEAGE).sum() / len(miles) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Histogram + KDE ──────────────────────────────────────────────────
bins_m = np.linspace(miles.min() - 5000, max(miles.max(), OUR_MILEAGE) + 10000, 16)
ax1.hist(miles, bins=bins_m, density=True, alpha=0.45, color='steelblue',
         edgecolor='white', label=f'Listings (n={len(miles)})')
kde_m = gaussian_kde(miles, bw_method=0.35)
xm = np.linspace(miles.min() - 10000, OUR_MILEAGE + 20000, 300)
ax1.plot(xm, kde_m(xm), color='navy', lw=2, label='KDE')
ax1.axvline(OUR_MILEAGE, color='red', ls='--', lw=2, label=f'Our car {OUR_MILEAGE:,} mi')
ax1.annotate(f'{OUR_MILEAGE:,} mi\n(exceeds all listings)',
             xy=(OUR_MILEAGE, 0),
             xytext=(OUR_MILEAGE - 30000, max(kde_m(xm)) * 0.7),
             fontsize=10, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
ax1.set_xlabel('Mileage')
ax1.set_ylabel('Density')
ax1.set_title('Mileage Distribution of Current Listings')
ax1.legend()
ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# ── Right: CDF ─────────────────────────────────────────────────────────────
sorted_m = np.sort(miles)
cdf_ym = np.arange(1, len(sorted_m) + 1) / len(sorted_m)
ax2.step(sorted_m, cdf_ym, where='post', color='steelblue', lw=2, label='Empirical CDF')
# Extend to our mileage
ax2.plot([sorted_m[-1], OUR_MILEAGE], [1.0, 1.0], color='steelblue', lw=2, ls=':')
ax2.axvline(OUR_MILEAGE, color='red', ls='--', lw=2)
ax2.plot(OUR_MILEAGE, 1.0, 'r*', ms=14, zorder=5, label=f'Our car ({OUR_MILEAGE:,} mi)')
ax2.annotate(f'100th %-ile\n(beyond all listings)',
             xy=(OUR_MILEAGE, 1.0),
             xytext=(OUR_MILEAGE - 40000, 0.65),
             fontsize=10, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
ax2.set_xlabel('Mileage')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('CDF of Listing Mileages')
ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax2.legend()

plt.tight_layout()
plt.savefig(DATA / 'fig2_mileage_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print(f"  MILEAGE ANALYSIS  (n = {len(miles)} listings)")
print(f"{'='*60}")
print(f"  Market range : {miles.min():,} – {miles.max():,} mi")
print(f"  Market mean  : {miles.mean():,.0f} mi")
print(f"  Market median: {np.median(miles):,.0f} mi")
print(f"  Our mileage  : {OUR_MILEAGE:,} mi")
print(f"  Percentile   : >{pct_mileage:.0f}th — our car exceeds ALL {len(miles)} current listings.")
print(f"  Interpretation: Cars at this mileage are rarely listed; most have")
print(f"                  been bought, scrapped, or aren't worth listing.")
print(f"{'='*60}")
"""))

# ============================================================================
# Cell 6 – Markdown: Price vs Mileage
# ============================================================================
cells.append(md(r"""
## 3. Price vs. Mileage: Value Assessment

A scatter plot with linear and polynomial regression quantifies the expected
price at 148 k miles by extrapolating the market trend. The residual (actual price
minus predicted price) tells us whether the Schererville car is over- or
under-priced relative to the current market.
"""))

# ============================================================================
# Cell 7 – Price vs Mileage scatter + regression (code)
# ============================================================================
cells.append(code(r"""
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter — size by inverse deal quality (bigger = worse deal)
deal_map = {'Great Deal': 1, 'Good Deal': 2, 'Fair Deal': 3, 'High Price': 4, None: 2.5}
sizes = np.array([deal_map.get(l.get('imv_deal_rating'), 2.5) for l in listings]) * 40

ax.scatter(miles, prices, s=sizes, c='steelblue', edgecolors='navy',
           alpha=0.75, zorder=3, label='Listings (n=15)')

# ── Linear regression ──────────────────────────────────────────────────────
m1, b1 = np.polyfit(miles, prices, 1)
xreg = np.linspace(miles.min() - 5000, OUR_MILEAGE + 5000, 200)
ax.plot(xreg, m1 * xreg + b1, 'k--', lw=1.5, label=f'Linear (R²={np.corrcoef(miles, prices)[0,1]**2:.3f})')

# ── Polynomial (deg 2) ────────────────────────────────────────────────────
coeffs2 = np.polyfit(miles, prices, 2)
p2 = np.poly1d(coeffs2)
r2_poly = 1 - np.sum((prices - p2(miles))**2) / np.sum((prices - prices.mean())**2)
ax.plot(xreg, p2(xreg), color='darkorange', ls='-.', lw=1.5,
        label=f'Quadratic (R²={r2_poly:.3f})')

# ── Confidence band (linear) ──────────────────────────────────────────────
n = len(miles)
x_mean = miles.mean()
y_pred_all = m1 * miles + b1
se = np.sqrt(np.sum((prices - y_pred_all)**2) / (n - 2))
sx = np.sqrt(np.sum((miles - x_mean)**2))
t_val = stats.t.ppf(0.975, df=n - 2)
ci = t_val * se * np.sqrt(1/n + (xreg - x_mean)**2 / (sx**2))
ax.fill_between(xreg, m1*xreg + b1 - ci, m1*xreg + b1 + ci,
                alpha=0.12, color='gray', label='95 % CI (linear)')

# ── Our car ────────────────────────────────────────────────────────────────
pred_lin = m1 * OUR_MILEAGE + b1
pred_poly = p2(OUR_MILEAGE)
ax.plot(OUR_MILEAGE, OUR_PRICE, 'r*', ms=18, zorder=5, label='Our car')
ax.annotate(f'Our car\n${OUR_PRICE:,} @ {OUR_MILEAGE:,} mi',
            xy=(OUR_MILEAGE, OUR_PRICE),
            xytext=(OUR_MILEAGE - 30000, OUR_PRICE + 1200),
            fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Predicted point
ax.plot(OUR_MILEAGE, pred_lin, 'kx', ms=12, mew=2, zorder=5)
ax.annotate(f'Predicted (linear)\n${pred_lin:,.0f}',
            xy=(OUR_MILEAGE, pred_lin),
            xytext=(OUR_MILEAGE - 35000, pred_lin + 500),
            fontsize=9, color='black',
            arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('Mileage')
ax.set_ylabel('Price ($)')
ax.set_title('Price vs. Mileage — 2011 BMW 328i Market')
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(DATA / 'fig3_price_vs_mileage.png', dpi=150, bbox_inches='tight')
plt.show()

delta_lin  = OUR_PRICE - pred_lin
delta_poly = OUR_PRICE - pred_poly
r2_lin = np.corrcoef(miles, prices)[0,1]**2

print(f"\n{'='*60}")
print(f"  PRICE-vs-MILEAGE REGRESSION  (n = {len(prices)} listings)")
print(f"{'='*60}")
print(f"  Linear model : price = {m1:.4f} × mileage + {b1:,.0f}")
print(f"     R²        : {r2_lin:.3f}")
print(f"     Predicted  : ${pred_lin:,.0f}  at {OUR_MILEAGE:,} mi")
print(f"     Residual   : ${delta_lin:+,.0f}  ({'OVER' if delta_lin > 0 else 'UNDER'}-priced)")
print()
print(f"  Quadratic     : R² = {r2_poly:.3f}")
print(f"     Predicted  : ${pred_poly:,.0f}  at {OUR_MILEAGE:,} mi")
print(f"     Residual   : ${delta_poly:+,.0f}  ({'OVER' if delta_poly > 0 else 'UNDER'}-priced)")
print()
best_model = 'Quadratic' if r2_poly > r2_lin else 'Linear'
best_pred  = pred_poly if r2_poly > r2_lin else pred_lin
best_delta = delta_poly if r2_poly > r2_lin else delta_lin
print(f"  ▸ Best-fit model: {best_model} (higher R²)")
print(f"  ▸ At {OUR_MILEAGE:,} mi the market predicts ≈${best_pred:,.0f}.")
if best_delta < 0:
    print(f"  ▸ Asking price ${OUR_PRICE:,} is ${abs(best_delta):,.0f} BELOW prediction — appears under-valued.")
else:
    print(f"  ▸ Asking price ${OUR_PRICE:,} is ${best_delta:,.0f} ABOVE prediction — slight premium.")
print(f"{'='*60}")
"""))

# ============================================================================
# Cell 8 – Markdown: NHTSA Complaint Landscape
# ============================================================================
cells.append(md(r"""
## 4. NHTSA Complaint Landscape: What Fails on These Cars

The 2011 BMW 328I has **678 total NHTSA complaints** — a substantial dataset.
Below we group complaints by component category, colour-coded by severity, to
reveal the dominant failure modes.
"""))

# ============================================================================
# Cell 9 – Component failure bar chart (code)
# ============================================================================
cells.append(code(r"""
# ── Normalise component names ──────────────────────────────────────────────
def norm_component(raw):
    """Map raw NHTSA component string to a tidy category."""
    raw = raw.upper().strip()
    # Take first comma-separated part then simplify
    first = raw.split(',')[0].strip()
    mapping = {
        'ENGINE AND ENGINE COOLING': 'ENGINE',
        'ENGINE': 'ENGINE',
        'POWER TRAIN': 'POWER TRAIN',
        'VEHICLE SPEED CONTROL': 'POWER TRAIN',
        'ELECTRICAL SYSTEM': 'ELECTRICAL',
        'AIR BAGS': 'AIR BAGS',
        'FUEL/PROPULSION SYSTEM': 'FUEL SYSTEM',
        'FUEL SYSTEM': 'FUEL SYSTEM',
        'SERVICE BRAKES': 'BRAKES',
        'SERVICE BRAKES, HYDRAULIC': 'BRAKES',
        'STEERING': 'STEERING',
        'SUSPENSION': 'SUSPENSION',
        'EXTERIOR LIGHTING': 'LIGHTING',
        'STRUCTURE': 'STRUCTURE',
        'VISIBILITY': 'VISIBILITY',
        'SEATS': 'SEATS',
        'WHEELS': 'WHEELS',
        'TIRES': 'TIRES',
    }
    for key, val in mapping.items():
        if key in first:
            return val
    return first[:20]  # Fallback: first 20 chars

comp_series = pd.Series([norm_component(c['components']) for c in c11])
comp_counts = comp_series.value_counts().head(12)
total = len(c11)

# Severity colour palette
severity_colors = {
    'ENGINE':      '#d32f2f',
    'POWER TRAIN': '#c62828',
    'FUEL SYSTEM': '#e65100',
    'ELECTRICAL':  '#ef6c00',
    'AIR BAGS':    '#f9a825',
    'STEERING':    '#2e7d32',
    'BRAKES':      '#1565c0',
    'SUSPENSION':  '#1565c0',
    'LIGHTING':    '#558b2f',
    'STRUCTURE':   '#6a1b9a',
    'VISIBILITY':  '#00838f',
    'SEATS':       '#4e342e',
    'WHEELS':      '#37474f',
    'TIRES':       '#37474f',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                gridspec_kw={'width_ratios': [2, 1]})

# ── Left: Horizontal bar chart ────────────────────────────────────────────
colors = [severity_colors.get(c, '#78909c') for c in comp_counts.index]
bars = ax1.barh(comp_counts.index[::-1], comp_counts.values[::-1],
                color=colors[::-1], edgecolor='white', height=0.65)
for bar, val in zip(bars, comp_counts.values[::-1]):
    pct = val / total * 100
    ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
             f'{val}  ({pct:.1f}%)', va='center', fontsize=9)
ax1.set_xlabel('Number of Complaints')
ax1.set_title(f'Top 12 Complaint Components — 2011 BMW 328I  (n={total})')
ax1.set_xlim(0, comp_counts.max() * 1.25)

# ── Right: Donut chart of major categories ────────────────────────────────
major = {
    'Engine / Powertrain': comp_series.isin(['ENGINE', 'POWER TRAIN', 'FUEL SYSTEM']).sum(),
    'Electrical': (comp_series == 'ELECTRICAL').sum(),
    'Safety (Airbags)': (comp_series == 'AIR BAGS').sum(),
    'Other': total,
}
major['Other'] -= (major['Engine / Powertrain'] + major['Electrical'] + major['Safety (Airbags)'])
donut_colors = ['#d32f2f', '#ef6c00', '#f9a825', '#90a4ae']
wedges, texts, autotexts = ax2.pie(
    major.values(), labels=major.keys(), colors=donut_colors,
    autopct='%1.1f%%', startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.45, edgecolor='white'))
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight('bold')
ax2.set_title('Complaint Category Breakdown')

plt.tight_layout()
plt.savefig(DATA / 'fig4_complaint_components.png', dpi=150, bbox_inches='tight')
plt.show()

eng_pct = comp_series.isin(['ENGINE', 'POWER TRAIN', 'FUEL SYSTEM']).sum() / total * 100
elec_pct = (comp_series == 'ELECTRICAL').sum() / total * 100
airbag_pct = (comp_series == 'AIR BAGS').sum() / total * 100
print(f"\n{'='*55}")
print(f"  COMPONENT FAILURE SUMMARY  (n = {total} complaints)")
print(f"{'='*55}")
print(f"  Engine / Powertrain / Fuel : {eng_pct:.1f}%  — dominant failure mode")
print(f"  Electrical                 : {elec_pct:.1f}%")
print(f"  Airbags (recall-driven)    : {airbag_pct:.1f}%")
print(f"  Top component: {comp_counts.index[0]} ({comp_counts.values[0]} complaints, {comp_counts.values[0]/total*100:.1f}%)")
print(f"{'='*55}")
"""))

# ============================================================================
# Cell 10 – Markdown: Failure Timeline by Mileage
# ============================================================================
cells.append(md(r"""
## 5. Failure Timeline by Mileage: When Things Break

Using the 210 complaints where mileage was extractable from the narrative text,
we build the mileage-at-failure distribution. The critical question: **what
percentage of all reported failures have already occurred before 148 k miles?**
"""))

# ============================================================================
# Cell 11 – Mileage-at-failure distribution (code)
# ============================================================================
cells.append(code(r"""
df_mi = pd.DataFrame(mileage_data)
mi_vals = df_mi['mileage'].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: Histogram + KDE ──────────────────────────────────────────────────
bins_fail = np.arange(0, mi_vals.max() + 25000, 25000)
ax1.hist(mi_vals, bins=bins_fail, density=True, alpha=0.45, color='steelblue',
         edgecolor='white', label=f'Complaints (n={len(mi_vals)})')
kde_fail = gaussian_kde(mi_vals, bw_method=0.3)
xf = np.linspace(0, mi_vals.max() + 10000, 400)
ax1.plot(xf, kde_fail(xf), color='navy', lw=2, label='KDE')
ax1.axvline(OUR_MILEAGE, color='red', ls='--', lw=2, label=f'Our car {OUR_MILEAGE:,} mi')
ax1.set_xlabel('Mileage at Complaint')
ax1.set_ylabel('Density')
ax1.set_title('Mileage-at-Failure Distribution (2011 328I)')
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# ── Right: CDF with component overlays ────────────────────────────────────
sorted_fail = np.sort(mi_vals)
cdf_fail = np.arange(1, len(sorted_fail)+1) / len(sorted_fail)
ax2.plot(sorted_fail, cdf_fail, color='steelblue', lw=2.5, label='All components')
ax2.axvline(OUR_MILEAGE, color='red', ls='--', lw=2)

pct_before = (mi_vals <= OUR_MILEAGE).sum() / len(mi_vals) * 100
ax2.plot(OUR_MILEAGE, pct_before/100, 'ro', ms=10, zorder=5)
ax2.annotate(f'{pct_before:.0f}% of failures\noccur ≤ {OUR_MILEAGE:,} mi',
             xy=(OUR_MILEAGE, pct_before/100),
             xytext=(OUR_MILEAGE - 70000, pct_before/100 - 0.18),
             fontsize=10, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

# Component-specific CDFs for top 3
top_components = df_mi['component'].apply(norm_component).value_counts().head(3).index.tolist()
comp_colors = {'ENGINE': '#d32f2f', 'ELECTRICAL': '#ef6c00', 'POWER TRAIN': '#1565c0'}
for comp in top_components:
    mask = df_mi['component'].apply(norm_component) == comp
    vals_c = np.sort(df_mi.loc[mask, 'mileage'].values)
    if len(vals_c) >= 3:
        cdf_c = np.arange(1, len(vals_c)+1) / len(vals_c)
        ax2.plot(vals_c, cdf_c, lw=1.5, ls='--',
                 color=comp_colors.get(comp, 'gray'),
                 label=f'{comp} (n={len(vals_c)})')

ax2.set_xlabel('Mileage at Complaint')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('CDF of Mileage-at-Failure')
ax2.legend(fontsize=8, loc='lower right')
ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.savefig(DATA / 'fig5_mileage_at_failure.png', dpi=150, bbox_inches='tight')
plt.show()

# Component-specific risk remaining
print(f"\n{'='*65}")
print(f"  MILEAGE-AT-FAILURE ANALYSIS  (n = {len(mi_vals)} complaints with mileage)")
print(f"{'='*65}")
print(f"  ★ {pct_before:.0f}% of all reported failures occur AT OR BEFORE {OUR_MILEAGE:,} miles.")
print(f"  ★ {100 - pct_before:.0f}% of failures remain AHEAD of us (>{OUR_MILEAGE:,} mi).")
print()
print(f"  Component-specific (% of that component's failures ≤ {OUR_MILEAGE:,} mi):")
for comp in top_components:
    mask = df_mi['component'].apply(norm_component) == comp
    vals_c = df_mi.loc[mask, 'mileage'].values
    pct_c = (vals_c <= OUR_MILEAGE).sum() / len(vals_c) * 100
    print(f"    {comp:<18}: {pct_c:.0f}% occurred by {OUR_MILEAGE:,} mi  "
          f"({(vals_c <= OUR_MILEAGE).sum()}/{len(vals_c)})")
print(f"{'='*65}")
"""))

# ============================================================================
# Cell 12 – Markdown: Complaint Trajectory
# ============================================================================
cells.append(md(r"""
## 6. Complaint Trajectory Over Time: Is the 328i Getting Worse?

NHTSA complaints are filed over the full lifespan of a model. A time-series
view helps distinguish genuine failure-rate increases from recall-awareness spikes.
We also compare the 2011 model (678 complaints) against the 2012 model (143).
"""))

# ============================================================================
# Cell 13 – Time series of complaints (code)
# ============================================================================
cells.append(code(r"""
from collections import Counter

def extract_year(date_str):
    """Parse MM/DD/YYYY and return year int."""
    try:
        parts = date_str.split('/')
        return int(parts[2])
    except Exception:
        return None

years_11 = [extract_year(c['dateComplaintFiled']) for c in c11]
years_12 = [extract_year(c['dateComplaintFiled']) for c in c12]
years_11 = [y for y in years_11 if y]
years_12 = [y for y in years_12 if y]

all_years = sorted(set(years_11 + years_12))
cnt_11 = Counter(years_11)
cnt_12 = Counter(years_12)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(all_years))
w = 0.38
bars1 = ax.bar(x - w/2, [cnt_11.get(y, 0) for y in all_years], w,
               color='steelblue', edgecolor='white', label=f'2011 328I (n={len(c11)})')
bars2 = ax.bar(x + w/2, [cnt_12.get(y, 0) for y in all_years], w,
               color='darkorange', edgecolor='white', label=f'2012 328I (n={len(c12)})')

# Recall annotations
recall_events = {
    2013: 'Recall 13V044\n(battery cable)',
    2014: 'Recall 14V176\n(VANOS bolts)',
    2017: 'Recalls 17V676/683\n(wiring + PCV)',
    2020: 'Recall 20V017\n(airbag)',
    2024: 'Recall 24V513\n(airbag)',
}
for yr, txt in recall_events.items():
    if yr in all_years:
        idx = all_years.index(yr)
        peak = cnt_11.get(yr, 0)
        ax.annotate(txt, xy=(idx - w/2, peak), xytext=(idx - w/2, peak + 22),
                    fontsize=7.5, ha='center', color='#b71c1c',
                    arrowprops=dict(arrowstyle='->', color='#b71c1c', lw=0.8))

ax.set_xticks(x)
ax.set_xticklabels(all_years, rotation=45, fontsize=9)
ax.set_xlabel('Year Complaint Filed')
ax.set_ylabel('Number of Complaints')
ax.set_title('NHTSA Complaints by Year Filed — 2011 vs 2012 BMW 328I')
ax.legend()

plt.tight_layout()
plt.savefig(DATA / 'fig6_complaint_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# Trend analysis — exclude recall-spike years for organic trend
non_recall_years = [y for y in all_years if y not in [2017, 2020, 2024]]
organic_11 = [cnt_11.get(y, 0) for y in non_recall_years if y >= 2014]
if len(organic_11) >= 3:
    slope, *_ = np.polyfit(range(len(organic_11)), organic_11, 1)
    trend = 'increasing' if slope > 1 else ('decreasing' if slope < -1 else 'stable')
else:
    trend = 'insufficient data'

print(f"\n{'='*60}")
print(f"  COMPLAINT TRAJECTORY")
print(f"{'='*60}")
print(f"  2011 328I total complaints : {len(c11)}")
print(f"  2012 328I total complaints : {len(c12)}")
print(f"  Ratio (2011 / 2012)        : {len(c11)/len(c12):.1f}×  (expected — 2011 sold far more)")
print(f"  Organic trend (excl. recall spikes): {trend}")
print(f"  Note: Large spikes in 2017 and 2024 correlate with recall awareness,")
print(f"        not sudden mechanical degradation.")
print(f"{'='*60}")
"""))

# ============================================================================
# Cell 14 – Markdown: Recall Risk Assessment
# ============================================================================
cells.append(md(r"""
## 7. Recall Risk Assessment

There are **7 NHTSA recall campaigns** covering the 2011 BMW 328I. The Carfax
for VIN WBAPK7C51BA820431 shows **0 open recalls** as of the most recent check,
suggesting all applicable campaigns have been remedied.
"""))

# ============================================================================
# Cell 15 – Recall table + visualization (code)
# ============================================================================
cells.append(code(r"""
# Build recall DataFrame
recall_rows = []
severity_map = {
    'Air Bags': 'CRITICAL',
    'Electrical System': 'HIGH',
    'Engine': 'HIGH',
    'Power Train': 'MEDIUM',
}

for r in recalls:
    comp = r['Component']
    sev = 'MEDIUM'
    for key, val in severity_map.items():
        if key.lower() in comp.lower():
            sev = val
            break
    if 'air bag' in comp.lower():
        sev = 'CRITICAL'

    recall_rows.append({
        'Campaign': r['NHTSACampaignNumber'],
        'Date': r['ReportReceivedDate'],
        'Component': comp,
        'Consequence': r['Consequence'][:120] + ('...' if len(r['Consequence']) > 120 else ''),
        'Severity': sev,
    })

df_recalls = pd.DataFrame(recall_rows)

# Sort by date
df_recalls['_date'] = pd.to_datetime(df_recalls['Date'], format='%m/%d/%Y')
df_recalls = df_recalls.sort_values('_date').drop(columns='_date').reset_index(drop=True)

# Colour-coded display
sev_colors = {'CRITICAL': '#d32f2f', 'HIGH': '#e65100', 'MEDIUM': '#f9a825'}

print("=" * 100)
print(f"  RECALL CAMPAIGNS — 2011 BMW 328I  ({len(df_recalls)} total)")
print("=" * 100)
for _, row in df_recalls.iterrows():
    print(f"\n  [{row['Severity']:>8}]  {row['Campaign']}  ({row['Date']})")
    print(f"            Component   : {row['Component']}")
    print(f"            Consequence : {row['Consequence']}")
print()
print("  ✓ Carfax shows 0 open recalls for VIN WBAPK7C51BA820431.")
print("    All applicable campaigns appear to have been completed.")
print("=" * 100)

# Severity bar chart
fig, ax = plt.subplots(figsize=(8, 3))
sev_counts = df_recalls['Severity'].value_counts().reindex(['CRITICAL', 'HIGH', 'MEDIUM']).fillna(0)
colors_sev = [sev_colors[s] for s in sev_counts.index]
ax.barh(sev_counts.index, sev_counts.values, color=colors_sev, edgecolor='white', height=0.5)
for i, (v, s) in enumerate(zip(sev_counts.values, sev_counts.index)):
    ax.text(v + 0.1, i, f'{int(v)} recall{"s" if v != 1 else ""}', va='center', fontsize=10)
ax.set_xlabel('Count')
ax.set_title('Recall Severity Breakdown (7 campaigns)')
ax.set_xlim(0, sev_counts.max() + 1.5)
plt.tight_layout()
plt.savefig(DATA / 'fig7_recalls.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ============================================================================
# Cell 16 – Markdown: Distribution Curve Summary
# ============================================================================
cells.append(md(r"""
## 8. Where This Car Lives on the Distribution Curve

A single summary figure placing our car along four key dimensions: price
percentile, failure-mileage percentile, remaining hazard by component, and
projected cost scenarios.
"""))

# ============================================================================
# Cell 17 – Summary quad-plot (code)
# ============================================================================
cells.append(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# ── Top-Left: Price percentile gauge ──────────────────────────────────────
ax = axes[0, 0]
pct_price_val = (prices <= OUR_PRICE).sum() / len(prices) * 100
# Simple horizontal gauge
ax.barh([0], [100], color='#e0e0e0', height=0.4, edgecolor='none')
ax.barh([0], [pct_price_val], color='#d32f2f', height=0.4, edgecolor='none')
ax.plot(pct_price_val, 0, 'v', color='#d32f2f', ms=15, zorder=5)
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel('Percentile')
ax.set_title(f'Price Position: {pct_price_val:.0f}th Percentile (${OUR_PRICE:,})', fontsize=12)
ax.text(50, -0.35, 'Cheaper ←                                              → More Expensive',
        ha='center', fontsize=9, color='gray')

# ── Top-Right: Mileage on NHTSA failure CDF ──────────────────────────────
ax = axes[0, 1]
sorted_f = np.sort(mi_vals)
cdf_f = np.arange(1, len(sorted_f)+1) / len(sorted_f)
ax.fill_between(sorted_f, 0, cdf_f, alpha=0.2, color='steelblue')
ax.plot(sorted_f, cdf_f, color='steelblue', lw=2)
pct_fail_before = (mi_vals <= OUR_MILEAGE).sum() / len(mi_vals) * 100
ax.axvline(OUR_MILEAGE, color='red', ls='--', lw=2)
ax.plot(OUR_MILEAGE, pct_fail_before/100, 'ro', ms=10, zorder=5)
ax.annotate(f'{pct_fail_before:.0f}% of failures\nalready behind us',
            xy=(OUR_MILEAGE, pct_fail_before/100),
            xytext=(OUR_MILEAGE - 60000, pct_fail_before/100 - 0.20),
            fontsize=10, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Mileage at Failure')
ax.set_ylabel('CDF')
ax.set_title(f'NHTSA Failure CDF — {pct_fail_before:.0f}% Occurred ≤ {OUR_MILEAGE:,} mi', fontsize=12)
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# ── Bottom-Left: Remaining hazard by component ───────────────────────────
ax = axes[1, 0]
comp_risk = {}
for comp in ['ENGINE', 'ELECTRICAL', 'POWER TRAIN', 'AIR BAGS', 'STEERING', 'BRAKES', 'FUEL SYSTEM']:
    mask = df_mi['component'].apply(norm_component) == comp
    vals_c = df_mi.loc[mask, 'mileage'].values
    if len(vals_c) >= 3:
        pct_after = (vals_c > OUR_MILEAGE).sum() / len(vals_c) * 100
        comp_risk[comp] = {'remaining_pct': pct_after, 'n': len(vals_c)}

if comp_risk:
    risk_sorted = sorted(comp_risk.items(), key=lambda x: x[1]['remaining_pct'], reverse=True)
    comp_names = [r[0] for r in risk_sorted]
    remaining  = [r[1]['remaining_pct'] for r in risk_sorted]
    occurred   = [100 - r for r in remaining]
    bar_colors_rem = ['#d32f2f' if r > 40 else '#ef6c00' if r > 25 else '#2e7d32' for r in remaining]

    y_pos = range(len(comp_names))
    ax.barh(y_pos, occurred, color='#81c784', edgecolor='white', height=0.6, label='Already occurred (≤148k)')
    ax.barh(y_pos, remaining, left=occurred, color=bar_colors_rem, edgecolor='white', height=0.6, label='Remaining risk (>148k)')

    for i, (occ, rem, info) in enumerate(zip(occurred, remaining, risk_sorted)):
        n = info[1]['n']
        ax.text(102, i, f'{rem:.0f}% ahead (n={n})', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_names, fontsize=9)
    ax.set_xlim(0, 130)
    ax.set_xlabel('% of Component Failures')
    ax.set_title(f'Remaining Failure Risk by Component (>{OUR_MILEAGE:,} mi)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)

# ── Bottom-Right: Cost projection ────────────────────────────────────────
ax = axes[1, 1]
scenarios = ['Best Case', 'Most Likely', 'Worst Case']
purchase   = [4200, 4400, 4800]
maint      = [1500, 4100, 6100]
total_cost = [p + m for p, m in zip(purchase, maint)]
resale     = [1500, 1200, 0]
net_cost   = [t - r for t, r in zip(total_cost, resale)]

colors_scen = ['#2e7d32', '#1565c0', '#d32f2f']
bars_sc = ax.bar(scenarios, net_cost, color=colors_scen, edgecolor='white', width=0.5)
for bar, nc, tc, rv in zip(bars_sc, net_cost, total_cost, resale):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'${nc:,.0f}\n(total ${tc:,.0f} − resale ${rv:,.0f})',
            ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Net Cost ($)')
ax.set_title('2-Year Net Cost-of-Ownership Scenarios', fontsize=12)
ax.set_ylim(0, max(net_cost) * 1.25)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

plt.suptitle('Dashboard — 2011 BMW 328i xDrive @ $4,800 / 148k mi',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(DATA / 'fig8_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ============================================================================
# Cell 18 – Markdown: Best / Worst Case
# ============================================================================
cells.append(md(r"""
## 9. Best Case / Worst Case Scenarios

Below we model three ownership scenarios over a 2–4 year horizon, estimating
total outlay, annual driving cost, and cost-per-mile. All maintenance estimates
are based on typical BMW N52 repair costs at independent shops.
"""))

# ============================================================================
# Cell 19 – Scenario analysis (code)
# ============================================================================
cells.append(code(r"""
# ── Scenario definitions ──────────────────────────────────────────────────

scenarios_detail = {
    'Best Case': {
        'purchase': 4200,
        'negotiated': 'Yes ($4,800 → $4,200)',
        'maintenance': {
            'Oil-pan gasket': 800,
            'Water pump / thermostat': 700,
        },
        'annual_routine': 600,       # oil changes, brakes, filters
        'years_owned': 4,
        'miles_per_year': 10000,
        'resale_at_end': 1500,       # ~190k mi
        'catastrophic': False,
    },
    'Most Likely': {
        'purchase': 4400,
        'negotiated': 'Yes ($4,800 → $4,400)',
        'maintenance': {
            'Water pump / thermostat': 800,
            'Oil-pan gasket': 800,
            'Front struts + mounts': 900,
            'Misc. (sensors, bushings)': 400,
        },
        'annual_routine': 800,
        'years_owned': 3,
        'miles_per_year': 10000,
        'resale_at_end': 1200,       # ~178k mi
        'catastrophic': False,
    },
    'Worst Case': {
        'purchase': 4800,
        'negotiated': 'No (paid asking)',
        'maintenance': {
            'Water pump / thermostat': 800,
            'Oil-pan gasket': 800,
            'Front struts + control arms': 1500,
            'Transfer case service': 500,
            'Surprise (cooling cascade / trans)': 2500,
        },
        'annual_routine': 800,
        'years_owned': 2,
        'miles_per_year': 8000,
        'resale_at_end': 0,          # total loss at ~165k
        'catastrophic': True,
    },
}

# ── Compute totals ─────────────────────────────────────────────────────────
rows = []
for name, s in scenarios_detail.items():
    total_maint = sum(s['maintenance'].values())
    routine = s['annual_routine'] * s['years_owned']
    total_outlay = s['purchase'] + total_maint + routine
    total_miles = s['miles_per_year'] * s['years_owned']
    net_cost = total_outlay - s['resale_at_end']
    cost_per_mile = net_cost / total_miles if total_miles else 0
    cost_per_month = net_cost / (s['years_owned'] * 12)

    rows.append({
        'Scenario': name,
        'Purchase': s['purchase'],
        'Repair / Maintenance': total_maint,
        'Routine (oil, brakes, etc.)': routine,
        'Total Outlay': total_outlay,
        'Resale Value': s['resale_at_end'],
        'Net Cost': net_cost,
        'Years Owned': s['years_owned'],
        'Total Miles': total_miles,
        'Cost / Mile': cost_per_mile,
        'Cost / Month': cost_per_month,
    })

df_scen = pd.DataFrame(rows).set_index('Scenario')

print("=" * 80)
print("  OWNERSHIP SCENARIO ANALYSIS — 2011 BMW 328i xDrive (Schererville)")
print("=" * 80)

for name, s in scenarios_detail.items():
    info = df_scen.loc[name]
    cat = '🟢' if name == 'Best Case' else ('🔵' if name == 'Most Likely' else '🔴')
    print(f"\n  {cat} {name.upper()}")
    print(f"     Purchase price   : ${info['Purchase']:,.0f}  ({s['negotiated']})")
    print(f"     Anticipated repairs:")
    for item, cost in s['maintenance'].items():
        print(f"        • {item:<38} ${cost:>6,}")
    print(f"     Routine maintenance ({s['years_owned']} yr) : ${info['Routine (oil, brakes, etc.)']:,.0f}")
    print(f"     Total outlay    : ${info['Total Outlay']:,.0f}")
    print(f"     Resale / salvage: ${info['Resale Value']:,.0f}")
    print(f"     ────────────────────────")
    print(f"     Net cost        : ${info['Net Cost']:,.0f}")
    print(f"     Ownership       : {s['years_owned']} yr × {s['miles_per_year']:,} mi/yr = {info['Total Miles']:,} mi")
    print(f"     Cost per mile   : ${info['Cost / Mile']:.2f}")
    print(f"     Cost per month  : ${info['Cost / Month']:,.0f}")
    if s['catastrophic']:
        print(f"     ⚠️  Assumes catastrophic failure at ~165k (total loss)")

print()
print("=" * 80)

# ── Waterfall / stacked-bar visualisation ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Stacked cost breakdown
scenario_names = list(scenarios_detail.keys())
purchase_vals  = [scenarios_detail[s]['purchase'] for s in scenario_names]
repair_vals    = [sum(scenarios_detail[s]['maintenance'].values()) for s in scenario_names]
routine_vals   = [scenarios_detail[s]['annual_routine'] * scenarios_detail[s]['years_owned'] for s in scenario_names]
resale_neg     = [-scenarios_detail[s]['resale_at_end'] for s in scenario_names]

x = np.arange(len(scenario_names))
w = 0.45
ax1.bar(x, purchase_vals, w, label='Purchase', color='#1565c0')
ax1.bar(x, repair_vals, w, bottom=purchase_vals, label='Repairs', color='#d32f2f')
bottoms2 = [p + r for p, r in zip(purchase_vals, repair_vals)]
ax1.bar(x, routine_vals, w, bottom=bottoms2, label='Routine Maint.', color='#ef6c00')
# Resale offset
for i, (rv, tot) in enumerate(zip(resale_neg, [b + r for b, r in zip(bottoms2, routine_vals)])):
    if rv < 0:
        ax1.bar(i, -rv, w, bottom=tot, color='#2e7d32', alpha=0.5,
                label='Resale credit' if i == 0 else None, hatch='//')
        ax1.text(i, tot + (-rv)/2, f'-${-rv:,}', ha='center', fontsize=8, color='white', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(scenario_names, fontsize=10)
ax1.set_ylabel('Total Cost ($)')
ax1.set_title('Cost Breakdown by Scenario')
ax1.legend(fontsize=8, loc='upper left')
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

# Right: Cost per mile comparison
cpm = [df_scen.loc[s, 'Cost / Mile'] for s in scenario_names]
colors_bar = ['#2e7d32', '#1565c0', '#d32f2f']
bars_cpm = ax2.bar(scenario_names, cpm, color=colors_bar, edgecolor='white', width=0.45)
for bar, c in zip(bars_cpm, cpm):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'${c:.2f}/mi', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cost per Mile ($)')
ax2.set_title('Net Cost per Mile by Scenario')
ax2.set_ylim(0, max(cpm) * 1.3)
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:.2f}'))

plt.tight_layout()
plt.savefig(DATA / 'fig9_scenario_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ============================================================================
# Cell 20 – Markdown: Conclusions
# ============================================================================
cells.append(md(r"""
## 10. Conclusions and Distributional Summary

A final synthesis of every statistical dimension analysed above.
"""))

# ============================================================================
# Cell 21 – Final summary (code)
# ============================================================================
cells.append(code(r"""
# Recompute key metrics for summary
pct_price_final = (prices <= OUR_PRICE).sum() / len(prices) * 100
pct_mileage_final = (miles <= OUR_MILEAGE).sum() / len(miles) * 100
pct_fail_final = (mi_vals <= OUR_MILEAGE).sum() / len(mi_vals) * 100

# Regression prediction (use best model from earlier)
m1, b1 = np.polyfit(miles, prices, 1)
coeffs2 = np.polyfit(miles, prices, 2)
p2 = np.poly1d(coeffs2)
r2_lin = np.corrcoef(miles, prices)[0,1]**2
r2_poly = 1 - np.sum((prices - p2(miles))**2) / np.sum((prices - prices.mean())**2)
if r2_poly > r2_lin:
    pred_final = p2(OUR_MILEAGE)
    model_name = 'Quadratic'
else:
    pred_final = m1 * OUR_MILEAGE + b1
    model_name = 'Linear'
delta_final = OUR_PRICE - pred_final

# Component remaining risk
risk_engine = risk_elec = risk_pt = 'N/A'
for comp_name, var_name in [('ENGINE', 'risk_engine'), ('ELECTRICAL', 'risk_elec'), ('POWER TRAIN', 'risk_pt')]:
    mask = df_mi['component'].apply(norm_component) == comp_name
    vals_c = df_mi.loc[mask, 'mileage'].values
    if len(vals_c) >= 3:
        pct_after = (vals_c > OUR_MILEAGE).sum() / len(vals_c) * 100
        if comp_name == 'ENGINE':
            risk_engine = f'{pct_after:.0f}%'
        elif comp_name == 'ELECTRICAL':
            risk_elec = f'{pct_after:.0f}%'
        else:
            risk_pt = f'{pct_after:.0f}%'

print("╔" + "═"*70 + "╗")
print("║" + " FINAL STATISTICAL SUMMARY ".center(70) + "║")
print("║" + " 2011 BMW 328i xDrive — VIN WBAPK7C51BA820431 ".center(70) + "║")
print("╠" + "═"*70 + "╣")
print("║" + "".center(70) + "║")
print("║" + f"  1. PRICE POSITION".ljust(70) + "║")
print("║" + f"     ${OUR_PRICE:,} sits at the {pct_price_final:.0f}th percentile of {len(prices)} listings.".ljust(70) + "║")
if delta_final < 0:
    print("║" + f"     {model_name} regression predicts ${pred_final:,.0f} → ${abs(delta_final):,.0f} UNDER-priced.".ljust(70) + "║")
else:
    print("║" + f"     {model_name} regression predicts ${pred_final:,.0f} → ${delta_final:,.0f} OVER-priced.".ljust(70) + "║")
print("║" + "".center(70) + "║")
print("║" + f"  2. MILEAGE POSITION".ljust(70) + "║")
print("║" + f"     {OUR_MILEAGE:,} mi exceeds ALL {len(miles)} listings (max {miles.max():,}).".ljust(70) + "║")
print("║" + f"     On the NHTSA failure curve: {pct_fail_final:.0f}% of complaints occur".ljust(70) + "║")
print("║" + f"     at or below {OUR_MILEAGE:,} mi (n={len(mi_vals)} w/ mileage data).".ljust(70) + "║")
print("║" + "".center(70) + "║")
print("║" + f"  3. KEY RISKS REMAINING".ljust(70) + "║")
print("║" + f"     Engine failures after 148k    : {risk_engine} of engine complaints".ljust(70) + "║")
print("║" + f"     Electrical after 148k         : {risk_elec} of electrical complaints".ljust(70) + "║")
print("║" + f"     Powertrain after 148k         : {risk_pt} of powertrain complaints".ljust(70) + "║")
print("║" + f"     Known unknowns: water pump, oil-pan gasket, front struts".ljust(70) + "║")
print("║" + "".center(70) + "║")
print("║" + f"  4. RECALLS".ljust(70) + "║")
print("║" + f"     7 campaigns total; Carfax shows 0 open → all remedied.".ljust(70) + "║")
print("║" + "".center(70) + "║")
print("║" + f"  5. COST-OF-OWNERSHIP".ljust(70) + "║")
ml_net = df_scen.loc['Most Likely', 'Net Cost']
ml_cpm = df_scen.loc['Most Likely', 'Cost / Mile']
ml_cpm_mo = df_scen.loc['Most Likely', 'Cost / Month']
print("║" + f"     Most-likely scenario: ${ml_net:,.0f} net over 3 years".ljust(70) + "║")
print("║" + f"     = ${ml_cpm:.2f}/mile  |  ${ml_cpm_mo:,.0f}/month".ljust(70) + "║")
print("║" + "".center(70) + "║")
print("║" + f"  6. BOTTOM LINE".ljust(70) + "║")
print("║" + f"     Strengths: 0 accidents, near-100% dealer service, known".ljust(70) + "║")
print("║" + f"     maintenance history, below-market price.".ljust(70) + "║")
print("║" + f"     Weaknesses: Highest mileage in the market, unknown status".ljust(70) + "║")
print("║" + f"     of water pump / oil-pan gasket / struts.".ljust(70) + "║")
print("║" + f"     ".ljust(70) + "║")
print("║" + f"     RECOMMENDATION: Conditional BUY at ≤$4,400 if a pre-".ljust(70) + "║")
print("║" + f"     purchase inspection confirms no active oil/coolant leaks".ljust(70) + "║")
print("║" + f"     and no excessive suspension play.".ljust(70) + "║")
print("║" + f"     CONFIDENCE: Moderate — the service history is a strong".ljust(70) + "║")
print("║" + f"     positive, but 148k mi on an N52 means the water pump".ljust(70) + "║")
print("║" + f"     and oil-pan gasket are statistically overdue.".ljust(70) + "║")
print("╚" + "═"*70 + "╝")
"""))

# ─── Assemble and write notebook ──────────────────────────────────────────────

nb = make_nb()
nb['cells'] = cells

out_path = r'C:\Users\lcawley\bridge\analysis.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
