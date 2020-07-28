# from pdb import set_trace
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict 

sns.set_palette('pastel')
folder = os.path.join(os.getcwd(), 'run-results')
def save_ax(ax, name):
    ax.set_ylim(0,)
    fig = ax.figure
    fig.savefig(f'{name}.png')
    plt.close(fig)

csvs = glob.glob(os.path.join(folder, '{*}.csv'))
aggregated_csvs = defaultdict(list) 
# find the differently-seeded runs for each configuration
for csv in csvs:
    agg_csv = re.compile(r'seed:(\d+),').sub('', csv)
    aggregated_csvs[agg_csv].append(csv)

# aggregate csvs
for pair in aggregated_csvs.items():
    (agg_csv, fpaths) = pair
    dfs = []
    for csv in fpaths:
        df = pd.read_csv(csv, dialect='unix')
        df['seed'] = int(re.compile(r'.*,seed:(\d+),.*').match(csv).group(1))
        dfs.append(df)
    df = pd.concat(dfs)

    var_name = 'dataset'
    id_var = 'epoch'
    value_name = 'loss'
    value_map = {'lossTrain': 'train', 'lossValid': 'validation'}
    ax = sns.lineplot(
        x=id_var,
        y=value_name,
        hue=var_name,
        data=df.rename(columns=value_map).melt(var_name=var_name, value_name=value_name, id_vars=[id_var], value_vars=value_map.values())
    )
    fname = agg_csv.replace('.csv', f'-{value_name}')
    save_ax(ax, fname)

    metrics = ['accValid']
    for metric in metrics:
        ax = sns.lineplot(
            x=id_var,
            y=metric,
            data=df
        )
        fname = agg_csv.replace('.csv', f'-{metric}')
        save_ax(ax, fname)
