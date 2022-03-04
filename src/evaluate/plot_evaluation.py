from argparse import ArgumentParser
from pathlib import Path
import seaborn as sns
import re
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from zmq import get_library_dirs


METRICS = ['totalL1', '0to10mmL1', '10to20mmL1', 'above20mmL1']
METRICS_TITLE = ['L1Loss', 'L1Loss in [0,10) mm', 'L1Loss in [10,20) mm', 'L1Loss above 20 mm']


def generate_bar_plot(df):
    df_grouped = df.groupby(['model', 'epoch', 'it_ot'])
    df_grouped_mean = df_grouped.mean()
    df_grouped_std = df_grouped.std()

    _, ax = plt.subplots(1, len(METRICS), figsize=(10, 5))
    for idx, metric in enumerate(sorted(METRICS)):
        df_mean = df_grouped_mean.get(metric).unstack()
        df_std = df_grouped_std.get(metric).unstack()
        df_mean.plot.box(ax=ax[idx], yerr=df_std, use_index=False, ylim=(0, None))

        ax[idx].set_title(METRICS_TITLE[idx], fontdict=dict(fontsize=9))

        x_axis_labels = list(df_mean.index)
        for i in range(len(x_axis_labels)):
            label = f"{x_axis_labels[i][0]} e{x_axis_labels[i][1]}"
            x_axis_labels[i] = label
        ax[idx].set_xticklabels(x_axis_labels)

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
    for idx, metric in enumerate(sorted(METRICS)):
        width = 0.6
        space = width * 1 # 2/3
        df_grouped = df.groupby(['model', 'epoch', 'it_ot'])[metric].apply(list)
        df_ot_grouped = df_grouped.loc[:, :, 'output/target']
        df_it_grouped = df_grouped.loc[:, :, 'input/target']

        bp_it = ax[idx].boxplot(df_it_grouped, positions=np.array(range(len(df_it_grouped)))*2.0-space, sym='', widths=width) 
        bp_ot = ax[idx].boxplot(df_ot_grouped, positions=np.array(range(len(df_ot_grouped)))*2.0+space, sym='', widths=width) 
        set_box_color(bp_it, '#D7191C')
        set_box_color(bp_ot, '#2C7BB6')

        ax[idx].set_title(METRICS_TITLE[idx], fontdict=dict(fontsize=9))

        # draw temporary red and blue lines and use them to create a legend
        ax[idx].plot([], c='#D7191C', label='input/target')
        ax[idx].plot([], c='#2C7BB6', label='output/target')
        ax[idx].legend(prop={'size': 7})

        xticks = [str(index) for index in df_ot_grouped.index]
        ax[idx].set_xticks(range(0, len(xticks) * 2, 2), xticks, rotation=90)

        plt.xlim(-2, len(xticks)*2)


def main(args):
    plot_bar = False
    df = pd.read_json(args.eval_path)\

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
