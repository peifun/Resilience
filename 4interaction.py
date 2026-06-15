# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

FS = 28
FS_TITLE = 32
AXIS_FONT = {'fontname': 'Times New Roman'}
BOX_COLORS = ['#2E86AB', '#D64045', '#8B5FBF']
LABELS = [r'$\beta$(FVC)', r'$\beta$(AT_CV)', r'$\beta$(FVC$\times$AT_CV)']

df = pd.read_csv(r"E:\Unmanaged Forest.csv")

def bootstrap_run(df, n_boot, seed=42):
    np.random.seed(seed)
    n_sample = int(len(df) * 0.7)
    coefs = {'FVC': [], 'AT_CV': [], 'Interaction': []}
    for i in range(n_boot):
        boot = df.sample(n=n_sample, replace=True, random_state=i)
        y = boot['Forest_TAC']
        x = boot['FOR_FVC'] - boot['FOR_FVC'].mean()
        m = boot['AT_CV'] - boot['AT_CV'].mean()
        X = sm.add_constant(pd.DataFrame({'FVC': x, 'AT_CV': m, 'Interaction': x * m}))
        m_boot = sm.OLS(y, X).fit()
        coefs['FVC'].append(m_boot.params['FVC'])
        coefs['AT_CV'].append(m_boot.params['AT_CV'])
        coefs['Interaction'].append(m_boot.params['Interaction'])
    return coefs

print("Running 200 bootstrap...")
res_200 = bootstrap_run(df, 200)
print("Running 300 bootstrap...")
res_300 = bootstrap_run(df, 300, seed=123)

for tag, res in [('200', res_200), ('300', res_300)]:
    print(f"\n=== Bootstrap {tag}  ===")
    for k in ['FVC', 'AT_CV', 'Interaction']:
        arr = np.array(res[k])
        print(f"  {k:15s}: mean={arr.mean():+.2f}, std={arr.std():.2f}, "
              f"CI=[{np.percentile(arr, 2.5):+.2f}, {np.percentile(arr, 97.5):+.2f}]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11), dpi=300)

def plot_box(ax, coefs, title, n_boot):
    box_data = [np.array(coefs['FVC']), np.array(coefs['AT_CV']), np.array(coefs['Interaction'])]
    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.45,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(visible=False))

    for patch, col in zip(bp['boxes'], BOX_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

    for i, (arr, col) in enumerate(zip(box_data, BOX_COLORS)):
        jitter = np.random.uniform(-0.08, 0.08, len(arr))
        ax.scatter(np.full(len(arr), i+1) + jitter, arr,
                   s=25, alpha=0.4, color=col, edgecolors='none')
        mu = arr.mean()
        ax.plot(i+1, mu, 'D', color='white', markeredgecolor='black',
                markeredgewidth=2, markersize=18)

        fs_annotate = FS + 12
        if i == 2:
            annotation_offset = -0.25
            ha_pos = 'right'
        else:
            annotation_offset = 0.35
            ha_pos = 'left'
        ax.text(i+1 + annotation_offset, mu,
                f'{mu:+.2f}', fontsize=fs_annotate, color=col,
                ha=ha_pos, va='center', fontweight='bold', **AXIS_FONT)

    ax.axhline(y=0, color='black', linewidth=1.2, linestyle='-')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABELS, fontsize=FS+6, **AXIS_FONT)
    ax.set_title(title, fontsize=FS_TITLE+6, fontweight='bold', **AXIS_FONT)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylabel('Regression coefficient', fontsize=FS+6, **AXIS_FONT)
    ax.tick_params(axis='y', labelsize=FS)

plot_box(ax1, res_200, '(a) Bootstrap 200 iterations', 200)
plot_box(ax2, res_300, '(b) Bootstrap 300 iterations', 300)
plt.tight_layout()
plt.close()

