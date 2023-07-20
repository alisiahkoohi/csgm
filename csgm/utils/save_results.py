import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import math

from .project_path import checkpointsdir, plotsdir
from .toy_dataset import quadratic

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 12}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def plot_toy_conditional_example_results(args, train_obj, val_obj, dataset,
                                         intermediate_samples,
                                         test_conditioning_input,
                                         noise_scheduler):
    print('Saving model and samples\n')
    print('Model directory: ', checkpointsdir(args.experiment), '\n')
    print('Plots directory: ', plotsdir(args.experiment), '\n')

    if len(train_obj) > 0 and len(val_obj) > 0:
        train_obj = np.array(train_obj).reshape(-1,
                                                args.max_epochs).mean(axis=0)
        # Plot training objective.
        fig = plt.figure(figsize=(7, 2.5))
        plt.plot(np.linspace(0, args.max_epochs, len(val_obj)),
                 np.array(val_obj),
                 color="green",
                 lw=1.0,
                 alpha=0.6,
                 label="validation")
        ax = plt.gca()
        ax.grid(True)
        plt.legend()
        plt.ylabel("Training objective")
        plt.xlabel("Epochs")
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(plotsdir(args.experiment), "obj.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

    for key, value in intermediate_samples.items():
        intermediate_samples[key] = np.stack(value)

    with torch.no_grad():
        true_samples = []

        for i, c_in in enumerate(test_conditioning_input):
            data = np.array(
                quadratic(n=args.val_batchsize*5, s=args.input_size))[...,
                                                                        0]
            data[:, [0, 1], :] = data[:, [1, 0], :]
            true_samples.append(data)
        true_samples = np.array(true_samples)
        true_samples = np.transpose(true_samples, (0, 2, 1, 3))

    for j, (key, value) in enumerate(intermediate_samples.items()):
        for i, sample in enumerate(value):
            fig = plt.figure(figsize=(7, 3), dpi=200)
            for k in range(64):
                plt.plot(
                    test_conditioning_input[j][
                        0, :].detach().cpu().numpy(),
                    sample[k, :],
                    linewidth=0.9,
                    color="#698C9E",
                    label='_nolegend_' if k > 0 else 'Predicted functions',
                    alpha=0.6)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.xlim([-3, 3])
            plt.ylim([-9, 9])
            plt.savefig(os.path.join(
                plotsdir(args.experiment),
                'posterior_test-data-{}_stage-{}.png'.format(j, i)),
                        format='png',
                        bbox_inches='tight',
                        dpi=400)
            plt.close(fig)



    fig = plt.figure(figsize=(7, 3), dpi=200)
    for k in range(100):
        plt.plot(true_samples[0][1, k, :],
                    true_samples[0][0, k, :],
                    linewidth=0.9,
                color="#C64D4D",
                label='_nolegend_' if k > 0 else 'True function samples',
                alpha=0.6)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim([-3, 3])
    plt.ylim([-9, 9])
    plt.savefig(os.path.join(plotsdir(args.experiment),
                                "true_samples.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)


    font_prop = matplotlib.font_manager.FontProperties(family='serif',
                                                        style='normal',
                                                        size=10)
    for j, (key, value) in enumerate(intermediate_samples.items()):
        for i, sample in enumerate(value):
            for k_idx, k in enumerate([8, 12, 16]):
                fig = plt.figure(figsize=(7, 3))
                ax = sns.kdeplot(sample[:, k],
                                    fill=True,
                                    bw_adjust=0.9,
                                    color="#698C9E",
                                    label='Predicted functions',
                                    )
                ax = sns.kdeplot(true_samples[0][0, :, k],
                                    fill=False,
                                    bw_adjust=0.9,
                                    color="#C64D4D",
                                    label='True function samples')
                plt.ylabel("Probability density function"),
                plt.xlim([-12, 12])
                plt.grid(True)
                plt.legend(loc='upper right', ncols=1, fontsize=10)
                plt.title(r"Conditional density, $x = %.2f$" % test_conditioning_input[j][0, k])
                plt.savefig(os.path.join(
                    plotsdir(args.experiment),
                    'marginal_test-data-{}_stage-{}-x-{}.png'.format(j, i, k)),
                            format='png',
                            bbox_inches='tight',
                            dpi=400)
                plt.close(fig)

