"""
Generate Cost x Feature Speed Figure

Reads expAH_feature_speed_test.csv and plots:
- X-axis: Cost multiplier kappa (0x, 1x, 1.5x, 2x, 2.5x, 3x)
- Y-axis: Mean OOS Net Sharpe Ratio (15 markets)
- Lines: Ridge_Fast (red dashed), Ridge_All (orange), Ridge_Slow (blue),
         MomentumSort (black thick)

Story: At zero costs, fast features have competitive gross SR. As costs rise,
fast features collapse while slow features stay flat. Visual proof of WHY
feature speed matters -- it's a cost interaction.

Output: paper_icaifv2/figures/cost_speed_crossover.pdf
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# Load expAH results
csv_path = 'scripts/concern_experiments/expAH_feature_speed_test.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Strategies: {df['Strategy'].unique()}")
print(f"CostMults: {sorted(df['CostMult'].unique())}")

# Filter to OOS only
oos = df[df['Period'] == 'OOS'].copy()

# Strategies to plot
# Ridge variants use rank targets now
strategies = {
    'Ridge_Fast': {'color': '#d62728', 'ls': '--', 'lw': 1.5, 'label': 'RT Ridge (Fast)'},
    'Ridge_All': {'color': '#ff7f0e', 'ls': '-', 'lw': 1.5, 'label': 'RT Ridge (All)'},
    'Ridge_SlowMed': {'color': '#2ca02c', 'ls': '-.', 'lw': 1.5, 'label': 'RT Ridge (Slow+Med)'},
    'Ridge_Slow': {'color': '#1f77b4', 'ls': '-', 'lw': 2.0, 'label': 'RT Ridge (Slow)'},
    'MomentumSort': {'color': 'black', 'ls': '-', 'lw': 2.5, 'label': 'Momentum Sort'},
}

# Cost multipliers (use 0.0 for gross = zero cost)
cost_mults = sorted(oos['CostMult'].unique())

# Add gross SR as the zero-cost point
# Gross SR is the same regardless of cost multiplier, so take from any cost level
fig, ax = plt.subplots(figsize=(6, 4))

for strat_name, style in strategies.items():
    srs = []
    xs = []

    # 0x point: gross SR
    gross_data = oos[oos['Strategy'] == strat_name]
    if len(gross_data) == 0:
        continue
    gross_sr = gross_data.groupby('Market')['Gross_SR'].first().mean()
    xs.append(0.0)
    srs.append(gross_sr)

    # Net SR at each cost multiplier
    for cm in cost_mults:
        cm_data = oos[(oos['Strategy'] == strat_name) & (oos['CostMult'] == cm)]
        if len(cm_data) == 0:
            continue
        mean_sr = cm_data['Net_SR'].mean()
        xs.append(cm)
        srs.append(mean_sr)

    ax.plot(xs, srs, color=style['color'], linestyle=style['ls'],
            linewidth=style['lw'], label=style['label'], marker='o',
            markersize=3, zorder=3 if strat_name == 'MomentumSort' else 2)

# Shade the region between fast and slow to highlight the "feature speed gap"
fast_data = oos[oos['Strategy'] == 'Ridge_Fast']
slow_data = oos[oos['Strategy'] == 'Ridge_Slow']
if len(fast_data) > 0 and len(slow_data) > 0:
    xs_shade = [0.0]
    fast_srs = [fast_data.groupby('Market')['Gross_SR'].first().mean()]
    slow_srs = [slow_data.groupby('Market')['Gross_SR'].first().mean()]
    for cm in cost_mults:
        f = fast_data[fast_data['CostMult'] == cm]['Net_SR'].mean()
        s = slow_data[slow_data['CostMult'] == cm]['Net_SR'].mean()
        if not (np.isnan(f) or np.isnan(s)):
            xs_shade.append(cm)
            fast_srs.append(f)
            slow_srs.append(s)
    ax.fill_between(xs_shade, fast_srs, slow_srs, alpha=0.1, color='gray',
                    label='Feature speed gap')

ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_xlabel(r'Cost Multiplier $\kappa$')
ax.set_ylabel('Mean OOS Net Sharpe Ratio')
ax.set_title('Feature Speed Under Transaction Costs')
ax.legend(loc='best', framealpha=0.9)
ax.set_xlim(-0.1, max(cost_mults) + 0.1)

# Tight layout
plt.tight_layout()

# Save
os.makedirs('paper_icaifv2/figures', exist_ok=True)
out_pdf = 'paper_icaifv2/figures/cost_speed_crossover.pdf'
out_png = 'paper_icaifv2/figures/cost_speed_crossover.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# Print summary table
print("\n=== Cost x Feature Speed Summary ===")
for strat_name in strategies:
    gross_data = oos[oos['Strategy'] == strat_name]
    if len(gross_data) == 0:
        continue
    gross_sr = gross_data.groupby('Market')['Gross_SR'].first().mean()
    net_3x = oos[(oos['Strategy'] == strat_name) & (oos['CostMult'] == 3.0)]['Net_SR'].mean()
    degradation = gross_sr - net_3x if not np.isnan(net_3x) else np.nan
    print(f"  {strat_name:20s}: Gross={gross_sr:.3f}  Net@3x={net_3x:.3f}  Degradation={degradation:.3f}")
