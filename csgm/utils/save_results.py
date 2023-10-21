import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import math
from scipy.signal import hilbert
import h5py

from .project_path import checkpointsdir, plotsdir

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 14}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def signal_to_noise(xhat, x):
    return -20.0 * math.log(
        np.linalg.norm(x - xhat) / np.linalg.norm(x)) / math.log(10.0)


def normalize_std(mu, sigma):
    analytic_mu = hilbert(mu, axis=1)
    return sigma * np.abs(analytic_mu) / (np.abs(analytic_mu)**2 +
                                          5e-1), analytic_mu


def find_index_closest_value(arr, value):
    absolute_diff = np.abs(arr - value)
    closest_index = np.argmin(absolute_diff)
    return closest_index


def plot_toy_conditional_example_results(args, train_obj, val_obj, x_test,
                                         sample_list):

    print('Saving model and samples\n')
    print('Model directory: ', checkpointsdir(args.experiment), '\n')
    print('Plots directory: ', plotsdir(args.experiment), '\n')

    if not os.path.exists(
            os.path.join(plotsdir(args.experiment), str(args.input_size))):
        os.mkdir(os.path.join(plotsdir(args.experiment), str(args.input_size)))

    fig = plt.figure("training logs", figsize=(7, 4))
    plt.plot(np.linspace(0, args.testing_epoch + 1, (args.testing_epoch + 1) *
                         len(train_obj) // (args.testing_epoch + 1)),
             train_obj,
             color="orange",
             alpha=1.0,
             label="training loss")
    plt.plot(np.linspace(0, args.testing_epoch + 1, args.testing_epoch + 1),
             val_obj,
             color="k",
             alpha=0.8,
             label="validation loss")
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)
    plt.title("Training and validation objective values")
    plt.ylabel("Objective function")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(plotsdir(args.experiment), "log.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    x_test = x_test.cpu().numpy()

    fig = plt.figure(figsize=(7, 3))
    for i in range(64):
        plt.plot(x_test[0, 1, :],
                 sample_list[i, :],
                 linewidth=1.4,
                 color="#698C9E",
                 label='_nolegend_' if i > 0 else 'Predicted functions',
                 alpha=0.6)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlim([-3, 3])
    plt.ylim([-9, 9])
    plt.savefig(os.path.join(plotsdir(args.experiment), str(args.input_size),
                             'posterior_samples.png'),
                format='png',
                bbox_inches='tight',
                dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 3), dpi=200)
    for i in range(64):
        plt.plot(x_test[0, 1, :],
                 x_test[i, 0, :],
                 linewidth=1.4,
                 color="#C64D4D",
                 label='_nolegend_' if i > 0 else 'True function samples',
                 alpha=0.6)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlim([-3, 3])
    plt.ylim([-9, 9])
    plt.savefig(os.path.join(plotsdir(args.experiment), str(args.input_size),
                             "true_samples.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    for val in [-1.0, 0.0, 1.0]:
        k = find_index_closest_value(x_test[0, 1, :], val)
        fig = plt.figure(figsize=(7, 3))
        ax = sns.kdeplot(sample_list[:, k],
                         fill=True,
                         bw_adjust=0.9,
                         color="#698C9E",
                         label='Predicted functions',
                         linewidth=1.4)
        ax = sns.kdeplot(x_test[:, 0, k],
                         fill=False,
                         bw_adjust=0.9,
                         color="#C64D4D",
                         label='True function samples')
        plt.ylabel("Probability density function")
        plt.xlim([-12, 12])
        plt.grid(True)
        plt.legend(loc='upper right', ncols=1, fontsize=14)
        plt.title(r"Conditional density, $x = %.2f$" % val)
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 str(args.input_size),
                                 'marginal_idx-{}.png'.format(k)),
                    format='png',
                    bbox_inches='tight',
                    dpi=400)
        plt.close(fig)

    if args.plot_multi_res:

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        file = h5py.File(
            os.path.join(checkpointsdir(args.experiment),
                         'collected_samples.h5'), 'r')

        for val in [-1.0, 0.0, 0.5]:
            fig = plt.figure(figsize=(7, 2.5))
            index = find_index_closest_value(x_test[0, 1, :], val)
            ax = sns.kdeplot(x_test[:, 0, index],
                             fill=True,
                             bw_adjust=0.9,
                             color="#000000",
                             label='True density')
            for input_idx, input_size in enumerate([20, 25, 30, 35, 40]):

                index = find_index_closest_value(
                    file['x_' + str(input_size)][...], val)

                ax = sns.kdeplot(
                    file[str(input_size)][:, index],
                    fill=False,
                    bw_adjust=0.9,
                    linewidth=1.4,
                    color=colors[input_idx],
                    label=str(input_size),
                )

            plt.ylabel("Density function")
            plt.xlim([-1.5, 4])
            plt.grid(True)
            plt.legend(loc='upper right', ncols=1, fontsize=13)
            plt.title(r"$y = %.2f$" % val)
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'marginal_val-{}.png'.format(val)),
                        format='png',
                        bbox_inches='tight',
                        dpi=400)
            plt.close(fig)

        fig = plt.figure(figsize=(7, 2))
        for i in range(64):
            plt.plot(
                x_test[0, 1, :],
                x_test[i, 0, :],
                linewidth=1.2,
                color="#000000",
                #  label='_nolegend_' if i > 0 else 'True functions',
                alpha=0.4)
        # plt.legend(fontsize=14)
        plt.grid(True)
        plt.title('True functions')
        plt.xlim([-3, 3])
        plt.ylim([-9, 9])
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'true_samples.png'),
                    format='png',
                    bbox_inches='tight',
                    dpi=400)
        plt.close(fig)

        for input_idx, input_size in enumerate([20, 25, 30, 35, 40]):

            fig = plt.figure(figsize=(7, 2))
            for i in range(64):
                plt.plot(file['x_' + str(input_size)][...],
                         file[str(input_size)][i, :],
                         linewidth=1.2,
                         color=colors[input_idx],
                         label='_nolegend_'
                         if i > 0 else 'grid size {}'.format(input_size),
                         alpha=0.4)
                plt.plot(file['x_' + str(input_size)][...],
                         file[str(input_size)][i, :],
                         '.',
                         markersize=2.2,
                         color=colors[input_idx],
                         alpha=0.4)
            plt.legend(fontsize=14)
            plt.title('Predicted functions')
            plt.grid(True)
            plt.xlim([-3, 3])
            plt.ylim([-9, 9])
            plt.savefig(os.path.join(
                plotsdir(args.experiment),
                'posterior_samples_grid-size-' + str(input_size) + '.png'),
                        format='png',
                        bbox_inches='tight',
                        dpi=400)
            plt.close(fig)

        file.close()


def plot_seismic_imaging_results(args, train_obj, val_obj, sample_list,
                                 true_image, rtm_image, test_idx):

    font = {'family': 'serif', 'style': 'normal', 'size': 17}
    matplotlib.rc('font', **font)

    print('\n Plots directory:', plotsdir(args.experiment), '\n')
    if not os.path.exists(
            os.path.join(plotsdir(args.experiment), str(test_idx))):
        os.mkdir(os.path.join(plotsdir(args.experiment), str(test_idx)))

    fig = plt.figure("training logs", figsize=(7, 4))
    plt.plot(np.linspace(0, args.testing_epoch + 1, (args.testing_epoch + 1) *
                         len(train_obj) // (args.testing_epoch + 1)),
             train_obj,
             color="orange",
             alpha=1.0,
             label="training loss")
    plt.plot(np.linspace(0, args.testing_epoch + 1, args.testing_epoch + 1),
             val_obj,
             color="k",
             alpha=0.8,
             label="validation loss")
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)
    plt.title("Training and validation objective values")
    plt.ylabel("Objective function")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(plotsdir(args.experiment), "log.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    samples_cum_mean = np.cumsum(sample_list, axis=0) / np.reshape(
        np.arange(1, sample_list.shape[0] + 1), (sample_list.shape[0], 1, 1))
    samples_mean = samples_cum_mean[-1:, ...]

    snr_list = []
    snr_list_cum_mean = []
    for j in range(sample_list.shape[0]):
        snr_list.append(
            signal_to_noise(sample_list[j, ...], true_image[0, ...]))
        snr_list_cum_mean.append(
            signal_to_noise(samples_cum_mean[j, ...], true_image[0, ...]))

    fig = plt.figure("snr per sample size", figsize=(4, 7))
    plt.semilogx(range(1, sample_list.shape[0] + 1),
                 snr_list_cum_mean,
                 color="#000000")
    # plt.title("Conditional mean SNR")
    plt.ylabel("Conditional mean signal-to-noise ratio (dB)")
    plt.xlabel("Number of posterior samples")
    plt.grid(True, which="both", axis="both")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "snr_vs_num_samples.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    spacing = [20, 12.5]
    extent = np.array([
        0.0, true_image.shape[1] * spacing[0],
        true_image.shape[2] * spacing[1], 0.0
    ]) / 1e3

    v0 = np.ones(true_image.shape[1:], dtype=np.float32) * 2.5
    v0 *= np.reshape(np.linspace(1.0, 4.5 / 2.5, v0.shape[1]),
                     (1, v0.shape[1]))
    s = 1.0 / v0
    s[:, :10] = 1.0 / 1.5
    m0 = s**2.0

    fig = plt.figure("m0", figsize=(7.68, 4.8))
    plt.imshow(
        m0.T,
        vmin=0.0,
        vmax=0.2,
        aspect=1,
        cmap="YlGnBu",
        resample=True,
        interpolation="lanczos",
        filterrad=1,
        extent=extent,
    )
    plt.title("Smooth background model")
    cb = plt.colorbar(fraction=0.03,
                      pad=0.01,
                      format=sfmt,
                      ticks=np.arange(0.0, 0.2, 0.05))
    # cb.set_label(label=r"$\frac{\mathrm{s}^2}{\mathrm{km}^2}$", fontsize=12)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "background.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure("x", figsize=(7.68, 4.8))
    plt.imshow(
        true_image[0, ...],
        vmin=-1.5e3,
        vmax=1.5e3,
        aspect=1,
        cmap="Greys",
        resample=True,
        interpolation="lanczos",
        filterrad=1,
        extent=extent,
    )
    plt.title("Ground-truth (unknown) image")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "true_model.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure("y", figsize=(7.68, 4.8))
    plt.imshow(
        rtm_image[0, ...] / 204,
        vmin=-9.5e3,
        vmax=9.5e3,
        aspect=1,
        cmap="Greys",
        resample=True,
        interpolation="lanczos",
        filterrad=1,
        extent=extent,
    )
    plt.title("Reverse-time migrated image")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "observed_data.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure("x", figsize=(7.68, 4.8))
    plt.imshow(
        samples_mean[0, ...],
        vmin=-1.5e3,
        vmax=1.5e3,
        aspect=1,
        cmap="Greys",
        resample=True,
        interpolation="lanczos",
        filterrad=1,
        extent=extent,
    )
    plt.title("Conditional mean estimate")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "conditional_mean.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure("x", figsize=(7.68, 4.8))
    plt.imshow(
        np.std(sample_list, axis=0),
        aspect=1,
        cmap="OrRd",
        resample=True,
        interpolation="kaiser",
        filterrad=1,
        extent=extent,
        norm=matplotlib.colors.LogNorm(vmin=6, vmax=90.0),
    )
    plt.title("Pointwise standard deviation")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt, ticks=[6, 10.0, 90.0])
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "pointwise_std.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure("x", figsize=(7.68, 4.8))
    plt.imshow(
        np.abs(true_image[0, ...] - samples_mean[0, ...]),
        vmin=10,
        vmax=13e1,
        aspect=1,
        cmap="magma",
        resample=True,
        interpolation="kaiser",
        filterrad=1,
        extent=extent,
    )
    plt.title("Prediction error")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "error.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    for ns in range(10):

        fig = plt.figure("x", figsize=(7.68, 4.8))
        plt.imshow(
            sample_list[ns, ...],
            vmin=-1.5e3,
            vmax=1.5e3,
            aspect=1,
            cmap="Greys",
            resample=True,
            interpolation="lanczos",
            filterrad=1,
            extent=extent,
        )
        plt.title("Posterior sample")
        plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
        plt.grid(False)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                                 'sample_' + str(ns) + '.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

    normalized_std, analytic_mu = normalize_std(samples_mean[0, ...],
                                                np.std(sample_list, axis=0))

    fig = plt.figure("x", figsize=(7.68, 4.8))
    plt.imshow(
        normalized_std,
        aspect=1,
        cmap="OrRd",
        resample=True,
        interpolation="kaiser",
        filterrad=1,
        extent=extent,
        norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=0.3),
    )
    plt.title("Normalized standard deviation")
    plt.colorbar(fraction=0.03, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "normalized_pointwise_std.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    font = {'family': 'serif', 'style': 'normal', 'size': 12}
    matplotlib.rc('font', **font)

    horiz_loc = [10, 245]
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(6, 12), sharey=True)

    # Loop to create the two plots
    for i, loc in enumerate(horiz_loc):
        ax = axs[i]

        ax.plot(
            samples_mean[0, :, loc][::-1],
            np.linspace(0.0, samples_mean.shape[2] * spacing[1] / 1e3,
                        samples_mean.shape[2])[::-1],
            color="#31BFF3",
            label="Conditional mean",
            linewidth=1.3,
        )
        ax.plot(
            true_image[0, :, loc][::-1],
            np.linspace(0.0, samples_mean.shape[2] * spacing[1] / 1e3,
                        samples_mean.shape[2])[::-1],
            "--",
            color="k",
            alpha=0.5,
            label="Ground-truth image",
            linewidth=1.3,
        )
        ax.fill_betweenx(
            np.linspace(0.0, samples_mean.shape[2] * spacing[1] / 1e3,
                        samples_mean.shape[2])[::-1],
            samples_mean[0, :, loc][::-1] -
            2.576 * np.std(sample_list, axis=0)[:, loc][::-1],
            samples_mean[0, :, loc][::-1] +
            2.576 * np.std(sample_list, axis=0)[:, loc][::-1],
            color="#FFAF68",
            alpha=0.8,
            label="%99 confidence interval",
            facecolor="#FFAF68",
            edgecolor="#FFAF68",
            linewidth=0.5,
            linestyle="solid",
        )
        ax.ticklabel_format(axis="y", style="sci", useMathText=True)
        ax.grid(True)
        ax.set_title("Profile at " + str(loc * spacing[0] / 1e3) + " km",
                     fontsize=14)
        ax.set_xlabel("Perturbation", fontsize=14)
        ax.set_xlim([-1400, 1300][::-1])
        ax.set_ylim([-0.3,
                     samples_mean.shape[2] * spacing[1] / 1e3 + 0.05][::-1])

    axs[0].set_ylabel("Depth (km)", fontsize=14)

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0)

    axs[1].legend(loc="upper left",
                  ncol=1,
                  fontsize=14,
                  bbox_to_anchor=(-0.7, 1.0))

    plt.savefig(os.path.join(plotsdir(args.experiment), str(test_idx),
                             "vertical_profile.png"),
                format="png",
                bbox_inches="tight",
                dpi=600,
                pad_inches=.02)
    plt.close(fig)

    samples_mean_snr = signal_to_noise(samples_mean[0, ...], true_image[0,
                                                                        ...])
    rtm_image_snr = signal_to_noise(rtm_image[0, ...] / 204, true_image[0,
                                                                        ...])

    with open(os.path.join(plotsdir(args.experiment), str(test_idx),
                           "snr-values.txt"),
              "w",
              encoding="utf-8") as f:
        f.write("SNR of conditional mean: " + str(samples_mean_snr) + "\n")
        f.write("SNR of RTM image: " + str(rtm_image_snr) + "\n")

        for j in range(sample_list.shape[0]):
            f.write("SNR of sample " + str(j) + ": " + str(snr_list[j]) + "\n")
