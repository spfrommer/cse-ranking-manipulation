import argparse
import typing as t
import numpy as np
import os

import tabulate

import file_utils, plot_utils

CATEGORY_N = 8

def table():
    models = file_utils.subdirectories('./out')
    model_results = {}
    for model in models:
        path = os.path.join('./out', model, 'adversarial', 'results.pkl')
        if not os.path.exists(path):
            continue
        results = file_utils.read_pickle(path)
        
        natural_adversarial_scores = []
        
        for _, category_res in results.items():
            promoted = category_res['promoted_product']
            xs = np.mean(category_res['natural_scores'][promoted])
            adversarial_score = np.mean(category_res['adversarial_scores'][promoted])
            natural_adversarial_scores.append((xs, adversarial_score))
            
        model_results[model] = natural_adversarial_scores

    models = list(model_results.keys())
    lookup_names = [plot_utils.model_lookup[m] for m in models]
    # Sort models according to lookup names
    models, lookup_names = zip(*sorted(zip(models, lookup_names), key=lambda x: x[1]))
    
    headers = ['Model', r'Mean $\Delta$ score', r'Mean $\Delta$ score %']
    table = []
    
    for i, model in enumerate(models):
        scores = model_results[model]
        
        delta_score, delta_percent = [], []
        
        for natural_score, adversarial_score in scores:
            delta_score.append(adversarial_score - natural_score)
            delta_percent.append(
                (adversarial_score - natural_score) / (CATEGORY_N - natural_score)
            )
        
        table.append(
            [lookup_names[i], np.mean(delta_score), np.mean(delta_percent) * 100]
        )
    
    print(tabulate.tabulate(
        table, headers=headers, tablefmt='latex_raw', floatfmt='.2f'
    ))
        
    

if __name__ == "__main__":
    table()