import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import seaborn as sns
import numpy as np
import math

from .project_path import checkpointsdir, plotsdir

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def plot_toy_example_results(args, train_obj, val_obj, dataset,
                             intermediate_samples, noise_scheduler):
    print('Saving model and samples\n')
    print('Model directory: ', checkpointsdir(args.experiment), '\n')
    print('Plots directory: ', plotsdir(args.experiment), '\n')

    if len(train_obj) > 0 and len(val_obj) > 0:
        train_obj = np.array(train_obj).reshape(-1,
                                                args.max_epochs).mean(axis=0)
        # Plot training objective.
        fig = plt.figure(figsize=(7, 2.5))
        plt.plot(np.linspace(0, args.max_epochs, len(train_obj)),
                 np.array(train_obj),
                 color="#000000",
                 lw=1.0,
                 alpha=0.8,
                 label="training")
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

    # Plot histogtams.
    intermediate_samples = np.stack(intermediate_samples)
    true_samples = dataset[:args.val_batchsize, :].cpu().numpy()
    for i, sample in enumerate(intermediate_samples):
        if intermediate_samples.shape[0] == 1:
            fig_name = "marginals_{}.png".format(args.max_epochs)
        else:
            fig_name = "marginals_{}.png".format(i * args.save_freq)

        # Create figure with subplots.
        fig, axes = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))
        for j in range(2):
            sns.histplot(sample[:, j], ax=axes[j], color="black")
            sns.histplot(true_samples[:, j], ax=axes[j], color="green")
            axes[j].set_title("Marginal {}".format(j + 1))
            axes[j].grid(True)
            axes[j].tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(plotsdir(args.experiment), fig_name),
                    format="png",
                    bbox_inches="tight",
                    dpi=200,
                    pad_inches=.02)
        plt.close(fig)

    # Plot samples.
    for i, sample in enumerate(intermediate_samples):
        if intermediate_samples.shape[0] == 1:
            fig_title = "Samples at epoch {}".format(args.max_epochs)
            fig_name = "samples_{}.png".format(args.max_epochs)
        else:
            fig_title = "Samples at epoch {}".format(i * args.save_freq)
            fig_name = "samples_{}.png".format(i * args.save_freq)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(sample[:, 0],
                    sample[:, 1],
                    s=2.5,
                    color="#000000",
                    alpha=0.8)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        ax = plt.gca()
        ax.grid(True)
        plt.title(fig_title)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(plotsdir(args.experiment), fig_name),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(true_samples[:, 0],
                true_samples[:, 1],
                s=2.5,
                color="green",
                alpha=0.8)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    ax = plt.gca()
    ax.grid(True)
    plt.title("True samples")
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(os.path.join(plotsdir(args.experiment), "true_samples.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    # # Set up the plot
    # fig, ax = plt.subplots(figsize=(10, 10))
    # scat = ax.scatter(true_samples[:, 0],
    #                   true_samples[:, 1],
    #                   s=2.5,
    #                   color="green",
    #                   alpha=0.8)
    # ax.set_xlim([-4, 4])
    # ax.set_ylim([-4, 4])

    # # Define the update function for the animation
    # def update(t):
    #     xt = noise_scheduler.add_noise(
    #         dataset,
    #         torch.tensor([t], device=dataset.device).repeat(dataset.shape[0]))
    #     scat.set_offsets(xt.cpu())
    #     ax.set_title("Timestep {}".format(t))
    #     return scat,

    # # Create the animation
    # ani = animation.FuncAnimation(fig,
    #                               update,
    #                               frames=range(len(noise_scheduler)),
    #                               interval=50,
    #                               blit=True)

    # ani.save(os.path.join(plotsdir(args.experiment), "forward_process.gif"),
    #          savefig_kwargs={
    #              'dpi': 400,
    #              'pad_inches': .02
    #          })
