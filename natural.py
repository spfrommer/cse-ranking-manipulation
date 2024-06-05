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

from attack import (
    run_target_and_evaluator,
    load_target,
    add_target_arguments,
)

import dataset as dataset

from helpers import file_utils, plot_utils, app_interface

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

np.set_printoptions(linewidth=200)

CATEGORY_N = 8

def out_original_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./out', args.target_model, 'natural', 'original', *file_path)

def plot_original_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./plots', args.target_model, 'natural', 'original', *file_path)

def out_rewritten_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./out', args.target_model, 'natural', 'rewritten', *file_path)

def plot_rewritten_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    return os.path.join('./plots', args.target_model, 'natural', 'rewritten', *file_path)


# Combined across different models
def plot_combined_path(*file_path: t.List[str]) -> str:
    return os.path.join('./plots', 'combined', *file_path)


    
fstat_label_lookup = {
    'product': 'Product name F-statistic',
    'doc': 'Document F-statistic',
    'contextpos': 'Context position F-statistic'
}

fstat_label_root_lookup = {
    'product': 'Product name',
    'doc': 'Document',
    'contextpos': 'Context position'
}



def original_doc_analysis(args: argparse.Namespace):
    if args.run_eval:
        if args.use_hosted_urls:
            assert 'online' in args.target_model, 'Model must be online to use URLs.'

        target = load_target(args)
        file_utils.ensure_created_directory(out_original_path(args))

        results = {}
        for category in tqdm.tqdm(dataset.get_categories()):
            products, doc_or_htmls = [], []
            for product, doc_or_html, _ in dataset.get_products(
                    category,
                    returned_doc='pages' if args.use_hosted_urls else 'content_truncate'
                ):

                products.append(product)
                doc_or_htmls.append(doc_or_html)
                
            user_query = dataset.user_query(category)
            
            if args.use_hosted_urls:
                urls = app_interface.upload_htmls_and_get_urls(doc_or_htmls)
                run_args = { 'urls': urls }
            else:
                run_args = { 'docs': doc_or_htmls }
            
                
            product_scores, context_orderings, responses = run_target_and_evaluator(
                target_chat=target,
                user_query=user_query,
                products=products,
                num_runs=args.num_runs,
                include_ordering_prompt=not args.no_ordering_prompt,
                shuffle_context_order=not args.no_shuffle_context_order,
                **run_args
            )
            
            results[category] = {
                'scores': product_scores,
                'context_orderings': context_orderings,
                'responses': responses,
            }
        
        file_utils.write_pickle(out_original_path(args, 'results.pkl'), results)
    
    original_doc_plot()

def original_doc_plot():
    results = file_utils.read_pickle(out_original_path(args, 'results.pkl'))
    file_utils.create_empty_directory(plot_original_path(args))
    for category, category_results in results.items():
        product_scores = category_results['scores']

        fig, ax = plt.subplots(figsize=(3, 4))
        
        labels, values = [], []
        for product, scores in product_scores.items():
            labels.append(product.brand)
            values.append(scores)
            
        labels, values = zip(*sorted(
            zip(labels, values), key=lambda x: x[0], reverse=True
        ))
        
        plot_utils.comparison_densityplot(
            values_a=values,
            values_b=None,
            label_a='',
            label_b='',
            categorical_labels=[plot_utils.truncate_string(l) for l in labels],
            continuous_label='Ranking score',
            ax=ax,
            legend=False
        )
        
        plt.savefig(plot_original_path(args, f'{category}.pdf'), bbox_inches='tight')
        plt.close()
        

def rewritten_doc_analysis(args: argparse.Namespace):
    if args.run_eval:
        target = load_target(args)
        file_utils.ensure_created_directory(out_rewritten_path(args))
        results = {}
        for category in tqdm.tqdm(dataset.get_categories()):
            products, docs = dataset.get_products_with_rewritten_docs(category) 
            product_n = len(products)
                
            user_query = dataset.user_query(category)
            
            # Maps (product_index, doc_index, context_position) to list of scores
            category_scores = {
                (product_index, doc_index, context_position): [] for
                product_index in range(product_n) for
                doc_index in range(product_n) for
                context_position in range(product_n)
            }
            category_selected_doc_indices = []
            category_responses = []

            for _ in tqdm.tqdm(list(range(args.num_permutations))):
                # nth element is document index for nth product in products
                selected_doc_indices = list(range(product_n))
                np.random.shuffle(selected_doc_indices)
                
                selected_docs = [docs[i][j] for i, j in enumerate(selected_doc_indices)]
            
                product_scores, context_orderings, responses = run_target_and_evaluator(
                    target_chat=target,
                    user_query=user_query,
                    products=products,
                    docs=selected_docs,
                    num_runs=1,
                    include_ordering_prompt=not args.no_ordering_prompt,
                    shuffle_context_order=True,
                )
                
                for (product, run_scores) in product_scores.items():
                    assert len(run_scores) == 1
                    score = run_scores[0]

                    orderings = context_orderings[product]
                    assert len(orderings) == 1
                    context_position = orderings[0]
                    
                    product_index = products.index(product)
                    doc_index = selected_doc_indices[product_index]
                    
                    key = (product_index, doc_index, context_position)
                    category_scores[key].append(score)
                
                category_selected_doc_indices.append(selected_doc_indices)
                category_responses.append(responses)
            
            results[category] = {
                'products': products,
                'docs': docs,
                'scores': category_scores,
                'selected_doc_indices': category_selected_doc_indices,
                'responses': category_responses
            }
        
        file_utils.write_pickle(out_rewritten_path(args, 'results.pkl'), results)

    rewritten_doc_heatmaps(args)
    rewritten_doc_fstatistic_scatters(args)
    
    file_utils.create_empty_directory(plot_combined_path('rewritten'))
    rewritten_doc_fstatistic_scatters(args)
    rewritten_doc_fstatistic_scatter_comparison()
    rewritten_doc_fstatistic_comparison()
    rewritten_doc_contextorder_comparison()
    
def rewritten_doc_heatmaps(args: argparse.Namespace):
    results = file_utils.read_pickle(out_rewritten_path(args, 'results.pkl'))
    file_utils.create_empty_directory(plot_rewritten_path(args, 'heatmaps'))
    
    for category, category_results in results.items():
        products = category_results['products']
        category_scores = category_results['scores']
        
        sum_score = np.zeros((CATEGORY_N, CATEGORY_N, CATEGORY_N))
        score_count = np.zeros((CATEGORY_N, CATEGORY_N, CATEGORY_N))
        for (product_index, doc_index, context_pos), scores in category_scores.items():
            sum_score[product_index, doc_index, context_pos] += np.sum(scores)
            score_count[product_index, doc_index, context_pos] += len(scores)
        
        all_axes = ['product', 'doc', 'contextpos']
        for ax1 in all_axes:
            for ax2 in all_axes:
                if ax1 == ax2:
                    continue
                rewritten_doc_heatmap_plot(
                    category, products, sum_score, score_count,
                    axes=[ax1, ax2], sort_by=ax1
                )
                rewritten_doc_heatmap_plot(
                    category, products, sum_score, score_count,
                    axes=[ax1, ax2], sort_by=ax2
                )

def rewritten_doc_heatmap_plot(
        category: str,
        products: t.List[Product],
        sum_score: np.ndarray, # 3d: product, doc, context_pos
        score_count: np.ndarray, # 3d: product, doc, context_pos
        axes=['product', 'doc'], # or 'context_pos'
        sort_by='product' # or 'doc'
    ):
    
    assert sort_by in axes
    
    file_utils.ensure_created_directory(plot_rewritten_path(args, 'heatmaps', category))
    
    # Reduce average score to two selected axes
    axis_lookup = { 'product': 0, 'doc': 1, 'contextpos': 2 }
    missing_axis = [ax for ax in ['product', 'doc', 'contextpos'] if ax not in axes][0]
    missing_axis_index = axis_lookup[missing_axis]
    projected_sum_score = np.sum(sum_score, axis=missing_axis_index)
    projected_score_count = np.sum(score_count, axis=missing_axis_index)
    average_score = np.divide(
        projected_sum_score, projected_score_count,
        out=np.zeros_like(projected_sum_score), where=projected_score_count!=0
    )
    
    # Get axes of average_score in correct order
    if axis_lookup[axes[0]] > axis_lookup[axes[1]]:
        average_score = np.swapaxes(average_score, 0, 1)
        projected_sum_score = np.swapaxes(projected_sum_score, 0, 1)
        projected_score_count = np.swapaxes(projected_score_count, 0, 1)

    nonsort_axis = 1 - axes.index(sort_by)
    
    # This is to sort off of averaged scores
    # Technically not correct because it doesn't take into account the weighting
    # average_score_nonsort_axis = np.mean(average_score, axis=nonsort_axis)
    
    average_score_nonsort_axis = np.divide(
        projected_sum_score.sum(axis=nonsort_axis),
        projected_score_count.sum(axis=nonsort_axis),
        out=np.zeros_like(projected_sum_score.sum(axis=nonsort_axis)),
        where=projected_score_count.sum(axis=nonsort_axis)!=0
    )
    
    sorted_indices = np.argsort(average_score_nonsort_axis)
    
    simultaneous_sort = 'product' in axes and 'doc' in axes
    
    labels = [plot_utils.truncate_string(p.brand, truncate_width=70) for p in products]
    ticklabels = {
        'product': labels,
        'doc': labels,
        'contextpos': [str(i) for i in range(CATEGORY_N)]
    }
    
    # Since axes[0] is first axis of average_score, which is the y axis of the heatmap
    xticklabels, yticklabels = ticklabels[axes[1]], ticklabels[axes[0]]
        
    if axes.index(sort_by) == 0 or simultaneous_sort:
        average_score = average_score[sorted_indices, :]
        yticklabels = [yticklabels[i] for i in sorted_indices]
    if axes.index(sort_by) == 1 or simultaneous_sort:
        average_score = average_score[:, sorted_indices]
        xticklabels = [xticklabels[i] for i in sorted_indices]
    
    
    fig, ax = plt.subplots(figsize=(4, 3))
    # Put the (0, 0) in the bottom left
    im = ax.imshow(np.flip(average_score, axis=0), cmap=plot_utils.truncated_plasma)
    
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    
    ax.set_xticklabels(xticklabels)
    # Make first y axis label at the bottom left
    ax.set_yticklabels(list(reversed(yticklabels))) 
    
    ax.set_xlabel(fstat_label_root_lookup[axes[1]])
    ax.set_ylabel(fstat_label_root_lookup[axes[0]])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Average ranking score', rotation=-90, va="bottom")
    
    filename = f'{axes[0]}_{axes[1]}_sorted_{sort_by}'
    fig_path = plot_rewritten_path(args, 'heatmaps', category, f'{filename}')
    plt.savefig(fig_path + '.pdf', bbox_inches='tight')
    plt.close()

def rewritten_doc_fstatistic_scatter_comparison():
    models = file_utils.subdirectories('./out')

    lookup_names = [plot_utils.model_lookup[m] for m in models]
    # Sort models according to lookup names
    models, lookup_names = zip(*sorted(zip(models, lookup_names), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(3.7, 3.7))
    max_x, max_y = 0, 0
    for i, (model, lookup_name) in enumerate(zip(models, lookup_names)):
        path = os.path.join('./out', model, 'natural', 'rewritten', 'results.pkl')
        if not os.path.exists(path):
            continue
        results = file_utils.read_pickle(path)
        
        f_stat_inputs = get_f_stat_inputs(results)
        category_f_stats = get_per_category_f_stats(f_stat_inputs)
        
        prod_fs, doc_fs = zip(*[f_stats[:2] for f_stats in category_f_stats.values()])
        
        # Trim out upper 5th percentile of outliers
        # Make sure to remove the same indices from both lists
        prod_fs = np.array(prod_fs)
        doc_fs = np.array(doc_fs)
        prod_outliers = np.where(prod_fs > np.percentile(prod_fs, 95))[0]
        doc_outliers = np.where(doc_fs > np.percentile(doc_fs, 95))[0]
        outliers = set(prod_outliers) | set(doc_outliers)
        prod_fs = np.delete(prod_fs, list(outliers))
        doc_fs = np.delete(doc_fs, list(outliers))
        
        ax.scatter(
            prod_fs, doc_fs,
            color=plot_utils.sel_colors[i], alpha=0.2, label=lookup_name
        )
        
        max_x = max(max_x, max(prod_fs))
        max_y = max(max_y, max(doc_fs))
    
    ax.set_xlabel(fstat_label_lookup['product'])
    ax.set_ylabel(fstat_label_lookup['doc'])
    
    max_f_stat = max(max_x, max_y)
    ax.plot([0, max_f_stat], [0, max_f_stat], 'k--')
    
    ax.legend(loc='upper right')
    
    ax.set_aspect('equal', adjustable='box')

    plt.savefig(
        plot_combined_path('rewritten', 'fstatistic_scatter.pdf'), bbox_inches='tight'
    )
    plt.close()

def rewritten_doc_fstatistic_comparison():
    models = file_utils.subdirectories('./out')
    model_results = {}
    for model in models:
        path = os.path.join('./out', model, 'natural', 'rewritten', 'results.pkl')
        if not os.path.exists(path):
            continue
        results = file_utils.read_pickle(path)
        
        f_stat_inputs = get_f_stat_inputs(results)
        category_f_stats = get_per_category_f_stats(f_stat_inputs)
        
        product_fstats, doc_fstats, context_pos_fstats = [], [], []
        for _, (product_f, doc_f, context_pos_f) in category_f_stats.items():
            product_fstats.append(product_f)
            doc_fstats.append(doc_f)
            context_pos_fstats.append(context_pos_f)
        
        model_results[model] = {
            'product': product_fstats,
            'doc': doc_fstats,
            'contextpos': context_pos_fstats
        }
    
    # Ignore models that don't have the rewritten analysis
    models = list(model_results.keys())
    lookup_names = [plot_utils.model_lookup[m] for m in models]
    # Sort models according to lookup names
    models, lookup_names = zip(*sorted(zip(models, lookup_names), key=lambda x: x[1]))
    
    all_fstats = ['product', 'doc', 'contextpos']
    
    model_medians = {
        model: [np.median(model_results[model][fstat]) for fstat in all_fstats]
        for model in models
    }
    model_25th = {
        model: [np.percentile(model_results[model][fstat], 25) for fstat in all_fstats]
        for model in models
    }
    model_75th = {
        model: [np.percentile(model_results[model][fstat], 75) for fstat in all_fstats]
        for model in models
    }
    
    max_y = 100
        
        
    fig, ax = plt.subplots(figsize=(4.1, 4))

    for i, model in enumerate(models):
        i_centered = i - (len(models) - 1) / 2
        x = np.arange(len(all_fstats)) + i_centered * 0.2
        y = model_medians[model]
        yerr = [
            np.subtract(y, model_25th[model]),
            np.subtract(model_75th[model], y)
        ]
        
        ax.errorbar(
            x, y, yerr=yerr,
            fmt='o', label=plot_utils.model_lookup[model],
            linewidth=2, color=plot_utils.sel_colors[i]
        )
    
    major_xticks = list(range(len(all_fstats)))
    ax.set_xticks(major_xticks, minor=False)
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticklabels([
        fstat_label_root_lookup[f].replace(' ', '\n') for f in all_fstats
    ])

    ax.set_xticks([i + 0.5 for i in major_xticks[:-1]], minor=True)
    ax.xaxis.grid(True, which='minor', alpha=0.6, linewidth=0.7)
    
    ax.set_xlim(min(major_xticks) - 0.5, max(major_xticks) + 0.5)
    ax.set_ylim(0, max_y)


    ax.set_ylabel('F-statistic')
    
    ax.legend(loc='upper left')

    plt.savefig(plot_combined_path('rewritten', 'fstatistic.pdf'), bbox_inches='tight')
    plt.close()

def rewritten_doc_fstatistic_scatters(args: argparse.Namespace):
    results = file_utils.read_pickle(out_rewritten_path(args, 'results.pkl'))
    file_utils.create_empty_directory(plot_rewritten_path(args, 'fstatistics'))
    
    f_stat_inputs = get_f_stat_inputs(results)
    category_f_stats = get_per_category_f_stats(f_stat_inputs)

    all_axes = ['product', 'doc', 'contextpos']
    for ax1 in all_axes:
        for ax2 in all_axes:
            if ax1 == ax2:
                continue
            f_statistic_scatter_plot(category_f_stats, axes=[ax1, ax2])
    
def f_statistic_scatter_plot(
        category_f_stats: t.Dict[str, t.Tuple[float, float, float]],
        axes=['product', 'doc']
    ):
    
    def get_points_for_axes(prod_f, doc_f, context_pos_f):
        lookup = { 'product': prod_f, 'doc': doc_f, 'contextpos': context_pos_f }
        return (lookup[axes[0]], lookup[axes[1]])
    
    fig, ax = plt.subplots()
    points = [get_points_for_axes(*f_stats) for f_stats in category_f_stats.values()]
    x, y = zip(*points)
    ax.scatter(x, y, color=plot_utils.sel_colors[2])
    
    ax.set_xlabel(fstat_label_lookup[axes[0]])
    ax.set_ylabel(fstat_label_lookup[axes[1]])
    
    max_f_stat = max(max(x), max(y))
    ax.plot([0, max_f_stat], [0, max_f_stat], 'k--')
    
    ax.set_aspect('equal', adjustable='box')
    
    fig_path = plot_rewritten_path(args, 'fstatistics', f'{axes[0]}_{axes[1]}')
    plt.savefig(fig_path + '.pdf', bbox_inches='tight')
    plt.close()

def get_f_stat_inputs(model_rewritten_results: t.Dict):
    f_stat_inputs = []
    
    for category, category_results in model_rewritten_results.items():
        products = category_results['products']
        category_scores = category_results['scores']
        
        sum_score = np.zeros((CATEGORY_N, CATEGORY_N, CATEGORY_N))
        score_count = np.zeros((CATEGORY_N, CATEGORY_N, CATEGORY_N))
        product_scores = [[] for _ in range(CATEGORY_N)]
        doc_scores = [[] for _ in range(CATEGORY_N)]
        context_pos_scores = [[] for _ in range(CATEGORY_N)]
        for (product_index, doc_index, context_pos), scores in category_scores.items():
            sum_score[product_index, doc_index, context_pos] += np.sum(scores)
            score_count[product_index, doc_index, context_pos] += len(scores)
            product_scores[product_index].extend(scores)
            doc_scores[doc_index].extend(scores)
            context_pos_scores[context_pos].extend(scores)

        f_stat_inputs.append(
            (category, products, product_scores, doc_scores, context_pos_scores)
        )
    
    return f_stat_inputs

def get_per_category_f_stats(
        f_stat_inputs: t.List[t.Tuple[
            str, # Category
            t.List[Product], # Products
            t.List[t.List[float]], # Product scores
            t.List[t.List[float]], # Doc scores
            t.List[t.List[float]], # Context position scores
        ]]
    ) -> t.Dict[str, t.Tuple[float, float, float]]: # Product, doc, context pos f stats

    category_f_stats = {}
    for (category, _, product_scores, doc_scores, context_pos_scores) in f_stat_inputs:
        product_f_stat = get_f_statistic(product_scores)
        doc_f_stat = get_f_statistic(doc_scores)
        context_pos_f_stat = get_f_statistic(context_pos_scores)
        category_f_stats[category] = (product_f_stat, doc_f_stat, context_pos_f_stat)
    
    return category_f_stats
    
def get_f_statistic(scores: t.List[t.List[float]]):
    # Arbitrary categories (just want to get outer grouping of scores)
    categories = [str(i) for i in range(len(scores))]

    X, Y = [], []
    for (product, sublist) in zip(categories, scores):
        for score in sublist:
            X.append(product)
            Y.append(score)

    data = {'X': X, 'Y': Y}
    model = ols('Y ~ X', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table['F']['X']
    
def rewritten_doc_contextorder_comparison():
    models = file_utils.subdirectories('./out')
    model_results = {}
    for model in models:
        path = os.path.join('./out', model, 'natural', 'rewritten', 'results.pkl')
        if not os.path.exists(path):
            continue
        results = file_utils.read_pickle(path)
        
        in_out_pos_pairs = []
        for _, category_results in results.items():
            for key, scores in category_results['scores'].items():
                _, _, context_pos = key
                for score in scores:
                    in_out_pos_pairs.append((context_pos, score))
        
        # Means and stds of score for each input context position
        in_out_pos_pairs = np.array(in_out_pos_pairs)
        
        means, stds = [], []
        for input_pos in range(CATEGORY_N):
            out_scores = in_out_pos_pairs[in_out_pos_pairs[:, 0] == input_pos][:, 1]
            
            means.append(np.mean(out_scores))
            stds.append(np.std(out_scores))
        
        model_results[model] = { 'means': means, 'stds': stds }
    
    # Ignore models that don't have the rewritten analysis
    models = list(model_results.keys())
    lookup_names = [plot_utils.model_lookup[m] for m in models]
    # Sort models according to lookup names
    models, lookup_names = zip(*sorted(zip(models, lookup_names), key=lambda x: x[1]))
    
    # Line plot for each model, with shaded error range
        
    fig, ax = plt.subplots()

    for i, model in enumerate(models):
        # Change input position from 0 being first and 7 laset
        # to 8 being first and 1 being last
        x = CATEGORY_N - np.arange(CATEGORY_N)
        y = model_results[model]['means']
        yerr = model_results[model]['stds']
        
        label = plot_utils.model_lookup[model]
        
        ax.plot(x, y, label=label, color=plot_utils.sel_colors[i])
        ax.fill_between(
            x, np.subtract(y, yerr), np.add(y, yerr),
            color=plot_utils.sel_colors[i], alpha=0.2
        )
    
    ax.set_xlabel('Input context position')
    ax.set_ylabel('Ranking score')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(1, CATEGORY_N + 1))
    ax.set_yticks(range(CATEGORY_N + 1))
    
    ax.set_xlim(1, CATEGORY_N)
    ax.set_ylim(0, CATEGORY_N)
    
    ax.legend()

    plt.savefig(plot_combined_path('rewritten', 'context_pos.pdf'), bbox_inches='tight')
    plt.close()

def add_common_arguments(parser):
    parser.add_argument('--run-eval', action='store_true')
    parser.add_argument('--no-ordering-prompt', action='store_true')

def add_original_doc_arguments(parser):
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--no-shuffle-context-order', action='store_true')
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--use-hosted-urls', action='store_true')
    
def add_rewritten_doc_arguments(parser):
    parser.add_argument('--rewritten', action='store_true')
    parser.add_argument('--num-permutations', type=int, default=80)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_target_arguments(parser)
    
    add_common_arguments(parser)
    add_original_doc_arguments(parser)
    add_rewritten_doc_arguments(parser)

    args = parser.parse_args()
    
    assert [args.original, args.rewritten].count(True) == 1

    if args.original:
        original_doc_analysis(args)
    elif args.rewritten:
        rewritten_doc_analysis(args)
