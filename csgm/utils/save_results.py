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
from .toy_dataset import ExamplesMGAN, quadratic

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 12}
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

    intermediate_samples = np.stack(intermediate_samples)
    true_samples = dataset[:args.val_batchsize*5, :].cpu().numpy()

    # Plot histogtams.
    # for i, sample in enumerate(intermediate_samples):
    #     if intermediate_samples.shape[0] == 1:
    #         fig_name = "marginals_{}.png".format(args.max_epochs)
    #     else:
    #         fig_name = "marginals_{}.png".format(i * args.save_freq)

    #     # Create figure with subplots.
    #     fig, axes = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))
    #     for j in range(2):
    #         sns.histplot(sample[:, j], ax=axes[j], color="black")
    #         sns.histplot(true_samples[:, j], ax=axes[j], color="green")
    #         axes[j].set_title("Marginal {}".format(j + 1))
    #         axes[j].grid(True)
    #         axes[j].tick_params(axis='both', which='major', labelsize=10)
    #     plt.savefig(os.path.join(plotsdir(args.experiment), fig_name),
    #                 format="png",
    #                 bbox_inches="tight",
    #                 dpi=200,
    #                 pad_inches=.02)
    #     plt.close(fig)

    # Calculate number of rows and columns of subplots.
    n_rows, n_cols = closest_squares(args.input_size)
    for i, sample in enumerate(intermediate_samples):
        # Create figure with subplots.
        fig, axes = plt.subplots(n_rows,
                                 n_cols,
                                 figsize=(n_cols * 6, n_rows * 4))
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                if n_rows == 1 or n_cols == 1:
                    ax_idx = idx
                else:
                    ax_idx = i, j
                if idx < args.input_size:
                    sns.histplot(sample[:, idx],
                                 ax=axes[ax_idx],
                                 color="black")
                    sns.histplot(true_samples[:, idx],
                                 ax=axes[ax_idx],
                                 color="green")
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 "marginals_{}.png".format(i)),
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

    # noise = torch.randn(dataset.shape)

    # # Define the update function for the animation
    # def update(t):

    #     t = torch.from_numpy(np.repeat(
    #         t, dataset.shape)).long()

    #     # Add noise to the data according to the noise schedule.
    #     xt = noise_scheduler.add_noise(dataset, noise, t)

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

        if args.dataset in ['mgan_4', 'mgan_5', 'mgan_6']:
            fwd_op = ExamplesMGAN(name=args.dataset)
            for i, c_in in enumerate(test_conditioning_input):
                c_in = c_in.repeat(args.val_batchsize*5, 1).unsqueeze(1)
                true_samples.append(fwd_op(c_in.cpu()).numpy())
            true_samples = np.array(true_samples)[..., 0, 0].T

        elif args.dataset == 'quadratic':
            for i, c_in in enumerate(test_conditioning_input):
                data = np.array(
                    quadratic(n=args.val_batchsize*5, s=args.input_size[0]))[...,
                                                                           0]
                data[:, [0, 1], :] = data[:, [1, 0], :]
                true_samples.append(data)
            true_samples = np.array(true_samples)
            true_samples = np.transpose(true_samples, (0, 2, 1, 3))

    if args.dataset in ['mgan_4', 'mgan_5', 'mgan_6']:
        font_prop = matplotlib.font_manager.FontProperties(family='serif',
                                                           style='normal',
                                                           size=10)
        for j, (key, value) in enumerate(intermediate_samples.items()):
            # Plotting conditional densities.
            for i, sample in enumerate(value):
                fig = plt.figure(figsize=(7, 7))
                ax = sns.kdeplot(sample[:, 0],
                                 fill=True,
                                 bw_adjust=0.9,
                                 color="#F4889A",
                                 label=r"Estimated, $x = $ %.2f" %
                                 test_conditioning_input[j])
                ax = sns.kdeplot(true_samples[:, j],
                                 fill=True,
                                 bw_adjust=0.9,
                                 color="#79D45E",
                                 label=r"True")
                for label in ax.get_xticklabels():
                    label.set_fontproperties(font_prop)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(font_prop)
                plt.ylabel("Probability density function",
                           fontproperties=font_prop)
                # plt.xlim([-0.045, 0.045])
                plt.grid(True)
                plt.legend()
                # plt.ylim([0, 125])
                plt.title("Conditional histograms")
                plt.savefig(os.path.join(
                    plotsdir(args.experiment),
                    'posterior_test-data-{}_stage-{}.png'.format(j, i)),
                            format='png',
                            bbox_inches='tight',
                            dpi=200)
                plt.close(fig)

    elif args.dataset == 'quadratic':
        # from IPython import embed; embed()
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
                for k_idx, k in enumerate([2, 12, 22]):
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
                    # for label in ax.get_xticklabels():
                    #     label.set_fontproperties(font_prop)
                    # for label in ax.get_yticklabels():
                    #     label.set_fontproperties(font_prop)
                    plt.ylabel("Probability density function"),
                                # fontproperties=font_prop)
                    plt.xlim([-12, 12])
                    plt.grid(True)
                    plt.legend(loc='upper right', ncols=1, fontsize=10)
                    # plt.ylim([0, 125])
                    plt.title(r"Conditional density, $x = %.2f$" % test_conditioning_input[j][0, k])
                    plt.savefig(os.path.join(
                        plotsdir(args.experiment),
                        'marginal_test-data-{}_stage-{}-x-{}.png'.format(j, i, k)),
                                format='png',
                                bbox_inches='tight',
                                dpi=400)
                    plt.close(fig)



def closest_squares(n):
    k = int(math.sqrt(n))
    while n % k != 0:
        k -= 1
    m = n // k
    return m, k
