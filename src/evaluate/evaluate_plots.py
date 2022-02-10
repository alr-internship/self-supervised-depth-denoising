from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd


def main(args):
    df = pd.read_csv(args.eval_path)

    # filter first N epochs
    df = df[df['epoch'] >= 5]

    # sort and reoder
    df.sort_values(by=['epoch', 'model'], inplace=True)
    df = df.pivot_table(values='mse', index='epoch', columns='model', aggfunc='first')

    # print line plot
    df.plot.line(figsize=(20, 8))
    plt.title("Depth-Denoising UNet")
    plt.ylabel("MSE (m)")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{args.eval_path.parent}/plt.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("eval_path", type=Path)
    main(parser.parse_args())
