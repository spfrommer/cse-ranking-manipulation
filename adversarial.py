import argparse
import typing as t
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

import tqdm

from _types import (
    Product
)

from models import Models

from attack import (
    get_adversarial_prompt,
    run_target_and_evaluator,
    load_target,
    load_attacker,
    add_target_arguments,
    add_attacker_arguments,
    add_tap_arguments,
    poison_doc
)

from natural import out_original_path as out_natural_path

import dataset as dataset

from helpers import file_utils, plot_utils, app_interface

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

CATEGORY_N = 8

def out_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./out', args.target_model, 'adversarial', *file_path)

def plots_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./plots', args.target_model, 'adversarial', *file_path)

# Combined across different models
def plot_combined_path(*file_path: t.List[str]) -> str:
    return os.path.join('./plots', 'combined', *file_path)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

truncated_plasma = truncate_colormap(plt.get_cmap('plasma'), 0.0, 0.96)



def adversarial_analysis(args: argparse.Namespace):
    if args.run_eval:
        natural_results = file_utils.read_pickle(out_natural_path(args, 'results.pkl'))
            
        target = load_target(args)

        file_utils.ensure_created_directory(out_path(args))
        adversarial_results = {}
        
        for category in tqdm.tqdm(dataset.get_categories()):
            assert category in natural_results
            
            user_query = dataset.user_query(category)
            
            natural_scores = natural_results[category]['scores']
            natural_context_orderings = natural_results[category]['context_orderings']
            natural_responses = natural_results[category]['responses']
            
            adversarial_prompt = None
            if args.transfer_attacks_from_model == 'none':
                promoted_product = min(
                    natural_scores, key=lambda prod: np.mean(natural_scores[prod])
                )
            else:
                source_results = file_utils.read_pickle(os.path.join(
                    './out', args.transfer_attacks_from_model,
                    'adversarial', 'results.pkl'
                ))
                promoted_product = source_results[category]['promoted_product']
                adversarial_prompt = source_results[category]['adversarial_prompt']

            
            products_doc_or_htmls = dataset.get_products(
                category,
                returned_doc='pages' if args.use_hosted_urls else 'content_truncate'
            )
            product_to_doc_or_html_lookup = {
                product: doc for product, doc, _ in products_doc_or_htmls
            }
            promoted_doc_or_html = product_to_doc_or_html_lookup[promoted_product]
            
            other_products = [
                p for p, _, _ in products_doc_or_htmls if p != promoted_product
            ]
            other_doc_or_htmls = [
                d for _, d, _ in products_doc_or_htmls if d != promoted_doc_or_html
            ]
            
            if args.use_hosted_urls:
                assert args.transfer_attacks_from_model != 'none'
                
                poisoned_html = app_interface.poison_html(
                    promoted_doc_or_html, adversarial_prompt
                )
                
                htmls_with_poison = [poisoned_html] + other_doc_or_htmls

                urls = app_interface.upload_htmls_and_get_urls(htmls_with_poison)
                
                run_args = { 'urls': urls }
            else:
                assert args.transfer_attacks_from_model == 'none'

                adversarial_prompt = get_adversarial_prompt(
                    user_query=user_query,
                    promoted_product=promoted_product,
                    promoted_doc=promoted_doc_or_html,
                    other_products=other_products,
                    other_docs=other_doc_or_htmls,
                    include_ordering_prompt=not args.no_ordering_prompt,
                    shuffle_context_order=not args.no_shuffle_context_order,
                    args=args
                )
                poisoned_doc = poison_doc(promoted_doc_or_html, adversarial_prompt)
                docs_with_poison = [poisoned_doc] + other_doc_or_htmls
                run_args = { 'docs': docs_with_poison }
            
            products = [promoted_product] + other_products

            
            product_scores, context_orderings, responses = run_target_and_evaluator(
                target_chat=target,
                user_query=user_query,
                products=products,
                num_runs=args.num_runs,
                include_ordering_prompt=not args.no_ordering_prompt,
                shuffle_context_order=not args.no_shuffle_context_order,
                **run_args
            )
            
            adversarial_results[category] = {
                'promoted_product': promoted_product,
                'promoted_doc': promoted_doc_or_html,
                'other_products': other_products,
                'other_docs': other_doc_or_htmls,
                'adversarial_prompt': adversarial_prompt,
                'natural_scores': natural_scores,
                'natural_context_orderings': natural_context_orderings,
                'natural_responses': natural_responses,
                'adversarial_scores': product_scores,
                'adversarial_context_orderings': context_orderings,
                'adversarial_responses': responses,
            }
        
        file_path = out_path(args, 'results.pkl')
        file_utils.write_pickle(file_path, adversarial_results)
    
    adversarial_plot(args)
    
    file_utils.create_empty_directory(plot_combined_path('adversarial'))
    adversarial_model_comparison()

def adversarial_plot(args: argparse.Namespace):
    all_results = file_utils.read_pickle(out_path(args, 'results.pkl'))
    file_utils.ensure_created_directory(plots_path(args))

    for (category, results) in all_results.items():
        fig, ax = plt.subplots(figsize=(3, 4))
        
        promoted_product = results['promoted_product']
        
        labels, original_scores, adversarial_scores = [], [], []
        for product in results['natural_scores'].keys():
            labels.append(product.brand)
            original_scores.append(results['natural_scores'][product])
            adversarial_scores.append(results['adversarial_scores'][product])
            
        labels, original_scores, adversarial_scores = zip(*sorted(
            zip(labels, original_scores, adversarial_scores),
            key=lambda x: x[0],
            reverse=True
        ))
        
        plot_utils.comparison_densityplot(
            values_a=original_scores,
            values_b=adversarial_scores,
            label_a='Natural',
            label_b='Adversarial',
            categorical_labels=labels,
            continuous_label='Ranking score',
            ax=ax,
            legend=True
        )
        
        promoted_label_index = labels.index(promoted_product.brand)
        
        tick_labels = ax.get_yticklabels()
        tick_labels[promoted_label_index].set_fontweight('bold')
        for i, label in enumerate(tick_labels):
            tick_labels[i] = plot_utils.truncate_string(label.get_text())
        ax.set_yticklabels(tick_labels)
        
        plt.savefig(plots_path(args, f'{category}.pdf'), bbox_inches='tight')
        plt.close()
        
def adversarial_model_comparison():
    plt.rcParams.update({'font.size': 13})

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
    
    fig, ax = plt.subplots()
    for i, model in enumerate(models):
        scores = model_results[model]
        natural_scores, adversarial_scores = zip(*scores)

        ax.scatter(
            natural_scores,
            adversarial_scores,
            color=plot_utils.sel_colors[i],
            alpha=0.2
        )
    
    for i, model in enumerate(models):
        scores = model_results[model]
        natural_scores, adversarial_scores = zip(*scores)
        
        natural_scores = np.round(natural_scores)
        xs, means, stds = [], [], []
        for n in range(CATEGORY_N + 1):
            scores = [a for a, n_ in zip(adversarial_scores, natural_scores) if n_ == n]
            if len(scores) == 0:
                continue
            xs.append(n)
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        
        means, stds = np.array(means), np.array(stds)
        
        color = plot_utils.sel_colors[i]
        linestyle = '-' if 'online' not in model else '--'

        ax.plot(
            xs,
            means,
            label=plot_utils.model_lookup[model],
            color=color,
            linewidth=2.5,
            linestyle=linestyle
        )
        ax.fill_between(
            xs,
            # Divide by two for less cluttered plot
            means - stds / 2,
            means + stds / 2,
            color=color,
            alpha=0.2
        )

        
    ax.set_xlim(-0.2, CATEGORY_N + 0.2)
    ax.set_ylim(-0.2, CATEGORY_N + 0.2)
    
    ax.plot([0, CATEGORY_N], [0, CATEGORY_N], 'k--')
    
    ax.set_xticks(range(CATEGORY_N + 1))
    ax.set_yticks(range(CATEGORY_N + 1))
    
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel('Natural average ranking score')
    ax.set_ylabel('Adversarial average ranking score')
    ax.legend()
    
    fig_path = plot_combined_path('adversarial', 'natural_vs_adversarial')
    plt.savefig(fig_path + '.pdf', bbox_inches='tight')
    plt.close()
        
    

def add_arguments(parser):
    parser.add_argument('--run-eval', action='store_true')
    parser.add_argument('--no-ordering-prompt', action='store_true')
    parser.add_argument('--no-shuffle-context-order', action='store_true')
    parser.add_argument('--use-hosted-urls', action='store_true')
    parser.add_argument('--transfer-attacks-from-model', type=str, default="none", choices=Models.keys())
    # For evaluating the adversarial attack once it's finished
    parser.add_argument('--num-runs', type=int, default=10)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_target_arguments(parser)
    add_attacker_arguments(parser)
    add_tap_arguments(parser)
    
    add_arguments(parser)

    args = parser.parse_args()
    
    adversarial_analysis(args)