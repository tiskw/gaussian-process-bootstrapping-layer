#!/usr/bin/env python3

"""
Visualize results of Gaussian process bootstrapping.
"""

import argparse
import math
import os
import pathlib

import pandas as pd
import matplotlib.pyplot as mpl


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--dirpath", type=str, default="results", help="path to results directory")
    parser.add_argument("--save", action="store_true", help="save figures under output directory")
    parser.add_argument("--output", type=str, default="figures", help="output directory")
    return parser.parse_args()


def read_result_tsv(path_tsv, length=300):
    """
    Parse TSV file and returns basename of file and list of test accuracy.
    """
    def is_test_row(line):
        """
        Returns true if the given line is a row of test score.
        """
        return line.startswith("Test: ")

    def get_test_accuracy(line):
        """
        Returns test accuracy based on the supposition that the given `line` is a test score row.
        """
        return float(line.split("\t")[0].split("=")[-1].strip())

    # Compute base name.
    basename = path_tsv.name[:-len(path_tsv.suffix)]

    # Read file and get test accuracy scores.
    scores = [get_test_accuracy(line) for line in path_tsv.open() if is_test_row(line)]

    # Normalize list length.
    if len(scores) < length: scores += [0.0] * (length - len(scores))
    else                   : scores = scores[:length]

    return (basename, scores)


def read_result_txt(path_txt, apply_sqrt=True):
    """
    Parse TXT file and returns a list of variance.
    """
    def is_variance_row(line):
        """
        Returns true if the given line is a row of test score.
        """
        return all(char in ".0123456789e+-" for char in line.strip())

    output = [float(line.strip()) for line in path_txt.open("rt") if is_variance_row(line)]

    if apply_sqrt: return list(map(math.sqrt, output))
    else         : return output


def plot_trends(scores, keys, header, args):
    """
    Plot test accuracy plotting for both with/without data augmentation.
    """
    # Plot accuracies without data augmentation.
    mpl.figure(figsize=(8, 5))
    mpl.title(f"{header}: without data augmentation")
    for key in keys:
        mpl.plot(100 * scores[key], "-o", label=key, lw=1, markersize=2, alpha=0.8)
    mpl.xlim(0, 150)
    mpl.ylim(60, 75)
    mpl.legend(loc="lower right", ncol=2)
    mpl.grid(linestyle="dotted")
    mpl.xlabel("Epoch")
    mpl.ylabel("Accuracy on CIFAR10 [%]")
    if args.save:
        filepath = "trend_" + header.lower().replace(" ", "_") + ".png"
        mpl.savefig(os.path.join(args.output, filepath))

    # Plot accuracies with data augmentation.
    mpl.figure(figsize=(8, 5))
    mpl.title(f"{header}: with data augmentation")
    for key in keys:
        mpl.plot(100 * scores[key + "_da"], "-o", label=key, lw=1, markersize=2, alpha=0.8)
    mpl.xlim(0, 300)
    mpl.ylim(75, 85)
    mpl.legend(loc="lower right", ncol=2)
    mpl.grid(linestyle="dotted")
    mpl.xlabel("Epoch")
    mpl.ylabel("Accuracy on CIFAR10 [%]")
    if args.save:
        filepath = "trend_" + header.lower().replace(" ", "_") + "_da.png"
        mpl.savefig(os.path.join(args.output, filepath))


def plot_topval(scores, keys, header, args):
    """
    Plot top accuracy of test data for both with/without data augmentation.
    """
    # Plot top accuracies without data augmentation.
    ks = keys
    xs = list(range(len(ks)))
    ys = [100 * max(scores[key]) for key in ks]
    mpl.figure(figsize=(8, 5))
    mpl.title(f"{header}: top scores without data augmentation")
    mpl.bar(xs, ys, color=[f"C{x}" for x in xs])
    for x, y in enumerate(ys):
        mpl.text(x - 0.25, y + 0.1, f"{y:.2f}", color=f"C{x}", fontweight="bold")
    mpl.xticks(xs, ks, rotation=20)
    mpl.ylim(70, 75)
    mpl.grid(linestyle="dotted")
    mpl.ylabel("Accuracy on CIFAR10 [%]")
    if args.save:
        filepath = "topscore_" + header.lower().replace(" ", "_") + ".png"
        mpl.savefig(os.path.join(args.output, filepath))

    # Plot top accuracies with data augmentation.
    ks = [k + "_da" for k in keys]
    xs = list(range(len(ks)))
    ys = [100 * max(scores[key]) for key in ks]
    mpl.figure(figsize=(8, 5))
    mpl.title(f"{header}: top scores with data augmentation")
    mpl.bar(xs, ys, color=[f"C{x}" for x in xs])
    for x, y in enumerate(ys):
        mpl.text(x - 0.25, y + 0.1, f"{y:.2f}", color=f"C{x}", fontweight="bold")
    mpl.xticks(xs, ks, rotation=20)
    mpl.ylim(80, 82.5)
    mpl.grid(linestyle="dotted")
    mpl.ylabel("Accuracy on CIFAR10 [%]")
    if args.save:
        filepath = "topscore_" + header.lower().replace(" ", "_") + "_da.png"
        mpl.savefig(os.path.join(args.output, filepath))


def plot_stdhist(std_plain, std_gpb_b, args):
    """
    Plot histogram of feature standard deviation.
    """
    mpl.figure(figsize=(8, 5))
    mpl.title("Histogram of feature standard deviation (N = 10,000)")
    mpl.hist(std_plain, bins=101, histtype="step", label="plain")
    mpl.hist(std_gpb_b, bins=101, histtype="step", label="gpb_b")
    mpl.grid(linestyle="dotted")
    mpl.xlabel("Standard deviation of features")
    mpl.ylabel("Frequency")
    mpl.legend()
    if args.save:
        filepath = "histogram_standard_deviation.png"
        mpl.savefig(os.path.join(args.output, filepath))


def main(args):
    """
    Main function.
    """
    # Path to the result directory.
    dirpath = pathlib.Path(args.dirpath)

    # Read test accuracy scores.
    scores = pd.DataFrame({name:score for name, score in map(read_result_tsv, sorted(dirpath.glob("*.tsv")))})

    # Target base names.
    keys1 = ["baseline", "gpb_tmx", "gpb_xmb", "gpb_tmb"]
    keys2 = ["baseline", "gpb_txx", "gpb_xmx", "gpb_xxb", "gpb_xmb", "gpb_txb", "gpb_tmx", "gpb_tmb"]

    # Plot figures and show them.
    plot_trends(scores, keys1, "Main results",       args)
    plot_trends(scores, keys2, "Exhaustive results", args)
    plot_topval(scores, keys2, "Exhaustive results", args)

    # Read test accuracy scores.
    std_plain = read_result_txt(dirpath / "variance_plain_da.txt", apply_sqrt=True)
    std_gpb_b = read_result_txt(dirpath / "variance_gpb_b_da.txt", apply_sqrt=True)

    # Plot figures and show them.
    plot_stdhist(std_plain, std_gpb_b, args)

    mpl.show()


if __name__ == "__main__":
    main(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
