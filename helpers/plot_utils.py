import string
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

truncated_plasma = truncate_colormap(plt.get_cmap('plasma'), 0.0, 0.96)

# sel_colors = ['#FA9E3B', '#C23D80', '#120789', '#226F54', '#353531']
sel_colors = ['#FA9E3B', '#C23D80', '#120789', '#226F54', '#3cd4e8']

model_lookup = {
    'gpt-3.5': 'GPT-3.5 Turbo',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'mixtral-8x22': 'Mixtral 8x22',
    'llama3-70b': 'Llama 3 70B',
    'llama3-sonar-large-online': 'Sonar Large Online',
}


def truncate_string(st, font_size=12, truncate_width=130):
    # Adapted from https://stackoverflow.com/questions/16007743/roughly-approximate-the-width-of-a-string-of-text-in-python
    size_to_pixels = (font_size / 12) * 16 * (6 / 1000.0) 
    truncate_width = truncate_width / size_to_pixels
    size = 0 # in milinches
    for i, s in enumerate(st):
        if s in 'lij|\' ': size += 37
        elif s in '![]fI.,:;/\\t': size += 50
        elif s in '`-(){}r"': size += 60
        elif s in '*^zcsJkvxy': size += 85
        elif s in 'aebdhnopqug#$L+<>=?_~FZT' + string.digits: size += 95
        elif s in 'BSPEAKVXY&UwNRCHD': size += 112
        elif s in 'QGOMm%W@–': size += 135
        else: size += 50

        if size >= truncate_width:
            return st[:max(0, i - 1)] + '...'

    return st

def comparison_densityplot(
        values_a: list[list[int]],
        values_b: Optional[list[list[int]]],
        label_a: str,
        label_b: str,
        categorical_labels: list[str],
        continuous_label: str,
        ax: plt.Axes,
        legend: bool=True,
        max_xaxis_value=8
    ):
    
    if values_b is None:
        values_b = [[] for _ in values_a]
        
    # Want a bunch of grouped bar charts, one right side up and one upside down
    # (values_a right side up and values_b up side down)
        
    # We should have one grouping per categorical label (outer list in values_a / values_b)
    
    assert (len(values_a) == len(categorical_labels))
    assert (len(values_b) == len(categorical_labels))
    
    n_groups = len(categorical_labels)
    n_values = len(values_a[0])
    
    valid_ranking_scores = range(max_xaxis_value + 1)
    
    ax.set_xlim(-0.5, max_xaxis_value + 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    
    ax.set_xticks(range(max_xaxis_value + 1))
    ax.set_yticks(range(n_groups))
    
    ax.set_yticklabels(categorical_labels)
    
    ax.set_xlabel(continuous_label)
    
    
    bar_width = 0.45
    opacity = 0.8

    for i, (va, vb) in enumerate(zip(values_a, values_b)):
        bar_heights_a = [va.count(i) / (2 * n_values) for i in valid_ranking_scores]
        bar_heights_b = [vb.count(i) / (2 * n_values) for i in valid_ranking_scores]
        
        for j, (ha, hb) in enumerate(zip(bar_heights_a, bar_heights_b)):
            if ha > 0:
                ax.add_patch(plt.Rectangle(
                    (j - bar_width / 2, i), bar_width, ha,
                    color=sel_colors[2], alpha=opacity
                ))
            if hb > 0:
                ax.add_patch(plt.Rectangle(
                    (j - bar_width / 2, i), bar_width, -hb,
                    color=sel_colors[0], alpha=opacity
                ))
        
        ax.axhline(y=i, color=(0.8, 0.8, 0.8), linewidth=1.0)
    
    # Manually make legend
    if legend:
        ax.add_patch(plt.Rectangle(
            (0, 0), 0, 0, color=sel_colors[2], alpha=opacity, label=label_a
        ))
        ax.add_patch(plt.Rectangle(
            (0, 0), 0, 0, color=sel_colors[0], alpha=opacity, label=label_b
        ))
        ax.legend(loc='lower left')