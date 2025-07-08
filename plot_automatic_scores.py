# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib
# import warnings
# import re
# import numpy as np
# from matplotlib.lines import Line2D
# import scipy.stats as stats
# import pandas as pd

# warnings.filterwarnings('ignore', message='FixedFormatter should only be used together with FixedLocator')

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['svg.fonttype'] = 'none'

# plt.rcParams.update({'font.size': 6, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial' })
# matplotlib.rcParams['axes.linewidth'] = 0.3


# # List of chemical questions and score types
# # chemical_questions = ['description', 'indication', 'pharmacodynamics', 'mechanism of action']
# chemical_questions = ['mechanism of interaction', 'management', ]
# # score_types = ['Human evaluation', 'cos sim', 'METEOR', 'Precision']
# # score_types = ['Human evaluation', 'cos sim', 'Precision', 'METEOR']

# score_types = ['cos sim', 'BLEU-1', 'METEOR']

# # Initialize figure
# fig, axes = plt.subplots(len(score_types), len(chemical_questions), figsize=(6.5, 5))
# # fig.suptitle('Comparison of Scores for Chemical Questions', fontsize=16)

# # Customize appearance
# flierprops = dict(marker='o', color='b', markersize=0.3)
# medianprops = dict(color="blue", linewidth=0.5)
# meanprops = dict(color="purple", linewidth=0.5)
# boxprops = dict(
#     linecolor='black', linewidth=0.4, width=.5,
#     showmeans=True, meanline=True, showcaps=True, 
#     showbox=True,
#     # showfliers=True,
#     # whis=[0, 100],
# )

# my_colors = {'GPT-4': (142/255.,153/255.,171/255.), 'InteractGPT': (183/255.,142/255.,114/255.)}

# # Create custom legend handles
# legend_elements = [
#     Line2D([0], [0], color=my_colors['GPT-4'], lw=5, label='GPT-4'),
#     Line2D([0], [0], color=my_colors['InteractGPT'], lw=5, label='InteractGPT')
# ]


# def cohen_d_from_samples(sample1, sample2, sided='two-sided'):
#     """
#     Computes Cohen's d effect size for a t-test from two sample lists.
    
#     Parameters:
#     - sample1, sample2: Lists or arrays of sample values for two groups.
#     - sided: 'one-sided' or 'two-sided', affects the interpretation but not the calculation.
    
#     Returns:
#     - Cohen's d value.
#     """
    
#     # Calculate means, standard deviations, and sample sizes for both samples
#     mean1 = np.mean(sample1)
#     mean2 = np.mean(sample2)
    
#     std1 = np.std(sample1, ddof=1)  # Standard deviation with Bessel's correction
#     std2 = np.std(sample2, ddof=1)
    
#     n1 = len(sample1)
#     n2 = len(sample2)
    
#     # Pooled standard deviation
#     pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
#     # Compute Cohen's d
#     d = (mean1 - mean2) / pooled_std

#     # print("Note: Effect size computation (Cohen's d) does not differ for one-sided vs two-sided tests.")
#     return d


# def get_stats(scores1, scores2, sided='two-sided'):
#     ttest_result = stats.ttest_ind(scores1, scores2, equal_var=False, alternative=sided)
#     tukey = stats.tukey_hsd(scores1, scores2)
#     print(ttest_result, tukey)
#     # import ipdb
#     # ipdb.set_trace()
#     t_statistic, p_value = ttest_result
#     df = ttest_result.df
#     ci95_low, ci95_high = ttest_result.confidence_interval()
#     effect_size = cohen_d_from_samples(scores1, scores2)
#     return t_statistic, p_value, df, ci95_low, ci95_high, effect_size


# def read_scores_and_preprocess(file_path):

#     remove_keys = {"management", "mechanism_interaction", "mean_scores", "advisory_terms", "classification_interaction"}
    
#     new_categories = {
#         "management": "management",
#         "mechanism_interaction": "mechanism of interaction"
#     }

#     with open(file_path, 'r') as f:
#         results = json.load(f)

#     scores = {chem_q: {st: [] for st in score_types} for chem_q in chemical_questions}
    
#     for smiles, result in results.items():
#         if smiles in remove_keys:
#             continue
        
#         for category in ["management", "mechanism_interaction"]:
#             category_data = result.get(category, {})
#             for key, value in category_data.items():
#                 if "similarity" in key:
#                     metric = 'cos sim'
#                 elif "meteor" in key:
#                     metric = 'METEOR'
#                 elif "bleu_1" in key:
#                     metric = 'BLEU-1'
#                 else:
#                     continue
                
#                 scores[new_categories[category]][metric].append(value)
    
#     return scores


    

# ttest_results = []
# all_scores = {t: {'GPT-4': [], 'InteractGPT': []} for t in score_types}
# alternative = 'two-sided'  # 'less' #
# gpt4_scores = read_scores_and_preprocess('/home/zhaoyang/project/drug-drug-interaction/eval_results/gpt_results.json')
# ours_scores = read_scores_and_preprocess('/home/zhaoyang/project/drug-drug-interaction/eval_results/ours.json')

# # Create box plots
# for i, question in enumerate(chemical_questions):
#     for j, score in enumerate(score_types):
#         # scores1 = get score of 'GPT-4'
#         # scores2 = get score of 'InteractGPT'
        
#         scores1 = gpt4_scores[question][score]
#         scores2 = ours_scores[question][score]
        
#         all_scores[score]['GPT-4'].extend(scores1)
#         all_scores[score]['InteractGPT'].extend(scores2)

#         ax = axes[j, i]
#         sns.boxplot(
#             data={'GPT-4': scores1, 'InteractGPT': scores2}, ax=ax, 
#             medianprops=medianprops, meanprops=meanprops, flierprops=flierprops, 
#             palette=my_colors,
#             # labels=['GPT-4', 'InteractGPT'] if i == j == 0 else [],
#             **boxprops)
#         # ax.set_title(f'{question} - {score}')
#         # ax.set_xticklabels(['GPT-4', 'InteractGPT'])

#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_tick_params(width=0.3)
#         ax.spines[['right', 'top']].set_visible(False)

#         if score == 'cos sim':
#             score = 'Semantic similarity'
#         if score == 'Precision':
#             score = 'BLEU'
#         if score == 'Human evaluation':
#             score = 'Human evaluation score'
#         if i == 0:
#             ax.set_ylabel(f'{score}')

#         # t test
#         if 'description' in question:
#             qq = 'overview'.capitalize()
#         else:
#             qq = question.capitalize()

#         t_statistic, p_value, df, ci95_low, ci95_high, effect_size = get_stats(scores1, scores2, alternative)
#         ttest_results.append([f'{qq}_{score}'.replace(' ', '_'), t_statistic, p_value, df, ci95_low, ci95_high, effect_size])

#         if j == 0:
#             if 'description' in question:
#                 ax.set_title('overview'.capitalize())
#             else:
#                 ax.set_title(question.capitalize())

#         if i == 0 and j == 0:
#             ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1.2, 1, 0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=2)

# # t test
# for score, res in all_scores.items():
#     scores1, scores2 = res['GPT-4'], res['InteractGPT']
#     if score == 'cos sim':
#         score = 'Semantic similarity'
#     if score == 'Precision':
#         score = 'BLEU'
#     if score == 'Human evaluation':
#         score = 'Human evaluation score'
#     t_statistic, p_value, df, ci95_low, ci95_high, effect_size = get_stats(scores1, scores2, alternative)
#     ttest_results.append([f'all_{score}'.replace(' ', '_'), t_statistic, p_value, df, ci95_low, ci95_high, effect_size])

# tt = pd.DataFrame(ttest_results, columns=['metric', 't_statistic', 'p_value', 'degree_of_freedom', 'ci95_low', 'ci95_high', 'Cohens_d_effect_size'])
# tt.to_csv('eval_results/similarity_scores/t_test_single_sided.csv', index=False)


# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig('eval_results/similarity_scores/scores_box_plot.png', dpi=400)
# plt.savefig('eval_results/similarity_scores/scores_box_plot.svg')
# plt.savefig('eval_results/similarity_scores/scores_box_plot.pdf')



import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

# -----------------------------------------------------------------------------
# Setup & Style
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="FixedFormatter should only be used together with FixedLocator",
)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "font.size": 6,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
})
matplotlib.rcParams["axes.linewidth"] = 0.3

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
chemical_questions = ["mechanism of interaction"]

score_types = [
    "cos sim",
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "METEOR",
    "ROUGE-1", "ROUGE-L",
]

metrics_to_plot = ["cos sim", "BLEU-1", "METEOR", "ROUGE-1"]

flierprops = dict(marker="o", markersize=0.3)
medianprops = dict(color="blue", linewidth=0.5)
meanprops = dict(color="purple", linewidth=0.5)
boxprops = dict(edgecolor="black", linewidth=0.4)

my_colors = {
    "GPT-4": (142/255., 153/255., 171/255.),
    "InteractGPT": (183/255., 142/255., 114/255.),
    "Apollo-MoE-7B": (85/255., 170/255., 85/255.),
    "MMed-Llama-3-8B": (170/255., 85/255., 170/255.),
}

legend_elements = [Line2D([0], [0], color=my_colors[k], lw=5, label=k) for k in my_colors]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _extract_score(val, idx=None):
    """
    抽取数值：
     - val 是 int/float：直接返回 float(val)
     - val 是能转 float 的字符串：返回 float(val)
     - val 是 dict，优先从 'score'、'fmeasure'、或 BLEU 的 'precisions' 列表里取
     - 否则返回 None
    """
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except:
        pass
    if isinstance(val, dict):
        if 'score' in val:
            return float(val['score'])
        if 'fmeasure' in val:
            return float(val['fmeasure'])
        if idx is not None and 'precisions' in val and len(val['precisions']) > idx:
            return float(val['precisions'][idx])
    return None

def load_scores(filepath):
    """Parse model metric JSON -> dict[question][metric] = list of floats."""
    raw = json.load(open(filepath, 'r'))
    out = {q: {m: [] for m in score_types} for q in chemical_questions}

    for rec in raw.values():
        if not isinstance(rec, dict):
            continue
        for key, val in rec.items():
            k = key.lower()

            # semantic similarity
            if 'semantic_similarity' in k or 'cosine' in k:
                score = _extract_score(val)
                if score is not None:
                    out['mechanism of interaction']['cos sim'].append(score)

            # BLEU-n
            elif 'bleu' in k:
                target = None
                if 'bleu-1' in k or 'bleu_1' in k or k.endswith('bleu1'):
                    target = 'BLEU-1'
                    idx = 0
                elif 'bleu-2' in k or 'bleu_2' in k or k.endswith('bleu2'):
                    target = 'BLEU-2'
                    idx = 1
                elif 'bleu-3' in k or 'bleu_3' in k or k.endswith('bleu3'):
                    target = 'BLEU-3'
                    idx = 2
                elif 'bleu-4' in k or 'bleu_4' in k or k.endswith('bleu4'):
                    target = 'BLEU-4'
                    idx = 3
                else:
                    idx = None
                if target:
                    score = _extract_score(val, idx=idx)
                    if score is not None:
                        out['mechanism of interaction'][target].append(score)

            # METEOR
            elif 'meteor' in k:
                score = _extract_score(val)
                if score is not None:
                    out['mechanism of interaction']['METEOR'].append(score)

            # ROUGE
            elif 'rouge' in k:
                score = None
                if isinstance(val, dict):
                    # ROUGE-L
                    if 'rougel' in k or 'rouge_l' in k or 'rouge-l' in k:
                        score = val.get('rougeL') or val.get('rouge_l') or val.get('fmeasure')
                    # ROUGE-1
                    elif 'rouge1' in k or 'rouge_1' in k or 'rouge-1' in k:
                        score = val.get('rouge1') or val.get('rouge_1') or val.get('fmeasure')
                    else:
                        score = val.get('fmeasure') or val.get('score')
                else:
                    try:
                        score = float(val)
                    except:
                        score = None

                if score is not None:
                    if 'rougel' in k or 'rouge_l' in k or 'rouge-l' in k:
                        out['mechanism of interaction']['ROUGE-L'].append(float(score))
                    elif 'rouge1' in k or 'rouge_1' in k or 'rouge-1' in k:
                        out['mechanism of interaction']['ROUGE-1'].append(float(score))

    return out

def one_sample_stats(values, popmean=0.0, alternative="two-sided"):
    """Return t-stat, p-val, dof, 95% CI, Cohen's d for a one-sample t-test."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return (np.nan,) * 6
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    t_res = stats.ttest_1samp(arr, popmean, alternative=alternative)
    t_stat, p_val = float(t_res.statistic), float(t_res.pvalue)
    dof = n - 1
    ci_low, ci_high = stats.t.interval(0.95, dof, loc=mean, scale=sd/np.sqrt(n))
    cohens = (mean - popmean) / sd if sd else np.nan
    return t_stat, p_val, dof, ci_low, ci_high, cohens

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
data_paths = {
    'GPT-4': 'eval_results/new/gpt_results.json',
    'InteractGPT': 'eval_results/new/ours.json',
    'Apollo-MoE-7B': 'eval_results/new/Apollo-MoE-7B_results_eval.json',
    'MMed-Llama-3-8B': 'eval_results/new/MMed-Llama-3-8B_results_eval.json'
}
models_data = {name: load_scores(path) for name, path in data_paths.items()}

# -----------------------------------------------------------------------------
# Plotting & Stats
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(7, 3))
fig.subplots_adjust(top=0.85)
fig.suptitle('Mechanism of Interaction', y=0.90, fontsize=8)

aggregate_scores = {m: [] for m in metrics_to_plot}
ttest_records = []

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    data = {m: models_data[m]['mechanism of interaction'][metric] for m in models_data}
    aggregate_scores[metric].extend(data['InteractGPT'])

    sns.boxplot(
        data=data, ax=ax, width=0.5, palette=my_colors,
        showmeans=True, meanline=True, showcaps=True,
        medianprops=medianprops, meanprops=meanprops,
        flierprops=flierprops, boxprops=boxprops
    )
    ax.set_xticks([])
    ax.set_ylabel(metric, fontsize=6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ttest_records.append([f"mech_int_{metric.replace(' ', '_')}", *one_sample_stats(data['InteractGPT'])])

for m, vals in aggregate_scores.items():
    ttest_records.append([f"all_{m.replace(' ', '_')}", *one_sample_stats(vals)])

out = Path('eval_results/new/similarity_scores')
out.mkdir(parents=True, exist_ok=True)

pd.DataFrame(ttest_records,
             columns=['metric', 't_stat', 'p_val', 'dof', 'ci_low', 'ci_high', 'cohens_d']) \
  .to_csv(out / 't_test_results.csv', index=False)

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.83),
           ncol=4, frameon=True, fontsize=6)
plt.tight_layout(rect=[0, 0, 1, 0.80])

for ext in ['png', 'svg', 'pdf']:
    plt.savefig(out / f'scores_box_plot.{ext}', dpi=400 if ext=='png' else None)
plt.close()

# Summary table
records = []

for name, data in models_data.items():
    for metric in score_types:
        vals = data['mechanism of interaction'][metric]
        if vals:
            mean_val = float(np.mean(vals))
            _, _, _, low, high, _ = one_sample_stats(vals)
            margin = (high - low) / 2
            mean_rounded = round(mean_val, 4)
            margin_rounded = round(margin, 4)
            records.append([name, metric, f"{mean_rounded} ± {margin_rounded}"])
        else:
            records.append([name, metric, np.nan])

summary = pd.DataFrame(records, columns=['model', 'metric', 'mean ± margin'])

summary.to_csv(out / 'mean_and_ci_results.csv', index=False)

print("=== Pivot Table (Means ± Margin) ===")
print(summary.pivot(index='model', columns='metric', values='mean ± margin'))

