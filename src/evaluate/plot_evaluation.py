from argparse import ArgumentParser
from distutils.ccompiler import new_compiler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import yaml

import pandas as pd


METRICS = ['totalL1', '0to10mmL1', '10to20mmL1', 'above20mmL1']
METRICS_TITLE = ['L1Loss', 'L1Loss in [0,10) mm', 'L1Loss in [10,20) mm', 'L1Loss above 20 mm']

def get_xticks(dir, trainer_ids):

    mapping = {}
    for trainer_id in trainer_ids:
        config_path = dir / trainer_id / 'config.yml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ds_config = config['dataset_config']
        nw_config = config['network_config']
        dd = ds_config['depth_difference_threshold'] if 'depth_difference_threshold' in ds_config else 0
        s = ds_config['scale_images']
        ic = nw_config['initial_channels']
        lr = nw_config['learning_rate']
        lt = nw_config['loss_type']
        o = nw_config['output_activation'] if 'output_activation' in nw_config else 'none'
        sk = nw_config['skip_connections'] if 'skip_connections' in nw_config else 0

        if lt == 'huber_loss':
            lt = 'h'
        elif lt == 'mean_l1_loss':
            lt = 'l1'
        elif lt == 'mean_l2_loss':
            lt = 'l2'

        if o == 'none':
            o = 'n'
        elif o == 'relu':
            o = 'r'

        mapping[trainer_id] = f"dd{dd}_s{s}_ic{ic}_lr{lr}_l{lt}_o{o}_sk{sk}"

    return mapping


def generate_bar_plot(df):
    df_grouped = df.groupby(['title', 'it_ot'])
    df_grouped_mean = df_grouped.mean()
    df_grouped_std = df_grouped.std()

    _, ax = plt.subplots(1, len(METRICS), figsize=(10, 5))
    for idx, metric in enumerate(METRICS):
        df_mean = df_grouped_mean.get(metric).unstack()
        df_std = df_grouped_std.get(metric).unstack()
        df_mean.plot.bar(ax=ax[idx], yerr=df_std, use_index=False, ylim=(0, None))

        ax[idx].set_title(METRICS_TITLE[idx], fontdict=dict(fontsize=9))
        ax[idx].set_xticklabels(df_mean.index)

        leg = ax[idx].legend(frameon=True, fontsize=8)
        leg.set_title(None)
        # leg.get_frame().set_alpha(None)
        # leg.get_frame().set_facecolor((1, 1, 1, 0.5))
        # leg.get_frame().set_edgecolor('black')
        # leg.get_frame().set_linewidth(0.5)


def get_box_plot(df: pd.DataFrame):
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    _, ax = plt.subplots(1, len(METRICS), figsize=(10, 5))
    for idx, metric in enumerate(METRICS):
        width = 0.6
        inner_space = width * 2/3
        outer_space = 2
        df_grouped = df.groupby(['title', 'it_ot'])[metric].apply(list)
        df_ot_grouped = df_grouped.loc[:, 'output/target']
        df_it_grouped = df_grouped.loc[:, 'input/target']
        print(df_it_grouped)

        bp_it = ax[idx].boxplot(df_it_grouped, positions=np.array(range(len(df_it_grouped)))*outer_space-inner_space, sym='', widths=width) 
        bp_ot = ax[idx].boxplot(df_ot_grouped, positions=np.array(range(len(df_ot_grouped)))*outer_space+inner_space, sym='', widths=width) 
        set_box_color(bp_it, '#D7191C')
        set_box_color(bp_ot, '#2C7BB6')

        ax[idx].set_title(METRICS_TITLE[idx], fontdict=dict(fontsize=9))

        # draw temporary red and blue lines and use them to create a legend
        ax[idx].plot([], c='#D7191C', label='input/target')
        ax[idx].plot([], c='#2C7BB6', label='output/target')
        ax[idx].legend(prop={'size': 7})

        xticks = df_ot_grouped.index
        ax[idx].set_xticks(range(0, len(xticks) * 2, 2), xticks, rotation=90)

        plt.xlim(-2, len(xticks)*2)


def main(args):
    plot_bar = False
    df = pd.read_json(args.eval_path, dtype={'model': str, 'epoch': str})\

    # unsqueeze metrics list to rows
    df = df.explode('metrics').reset_index()

    # metrics dict to columns
    df = df.drop('metrics', axis=1).join(pd.DataFrame(df.metrics.values.tolist())).drop('index', axis=1)
    df.rename(columns={'it': 'input/target', 'ot': 'output/target'}, inplace=True)

    df = df.set_index(['model', 'epoch'])

    df = df.stack().to_frame(name='metrics')
    df.index.set_names('it_ot', level=2, inplace=True)

    df = df['metrics'].apply(pd.Series)
    df = df.reset_index()

    xticks = get_xticks(args.eval_path.parent, df['model'].to_list())
    df.insert(0, 'title', [xticks[t] for t in df['model']])
    df = df.drop(['model', 'epoch'], axis=1)

    if plot_bar:
        generate_bar_plot(df)
    else:
        get_box_plot(df)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    # plt.show()
    plt.savefig(f"{args.eval_path.parent}/plt.png")  # , bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("eval_path", type=Path)
    main(parser.parse_args())
