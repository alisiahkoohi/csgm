import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import h5py

from csgm import NoiseScheduler, ConditionalScoreModel1D
from csgm.utils import (make_experiment_name, plot_seismic_imaging_results,
                        plot_toy_conditional_example_results, checkpointsdir,
                        plotsdir, query_experiments, CustomLRScheduler,
                        save_exp_to_h5, load_exp_from_h5, quadratic)

CONFIG_FILE = 'toy_example_conditional_quadratic.json'


def train(args):
    args.experiment = make_experiment_name(args)

    # Random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda > -1:
        device = torch.device('cuda:' + str(args.cuda))
    else:
        device = torch.device('cpu')

    # Initialize the network that will learn the score function.
    model = ConditionalScoreModel1D(
        modes=args.modes,
        hidden_dim=args.hidden_dim,
        nlayers=args.nlayers,
        nt=args.nt,
    ).to(device)

    # Setup the batch index generator.
    train_idx_loader = torch.utils.data.DataLoader(range(args.num_train //
                                                         args.batchsize),
                                                   batch_size=1,
                                                   shuffle=True,
                                                   drop_last=True)

    # Forward diffusion process noise scheduler.
    noise_scheduler = NoiseScheduler(nt=args.nt,
                                     beta_schedule=args.beta_schedule,
                                     device=device)

    # Optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Setup the learning rate scheduler.
    scheduler = CustomLRScheduler(optimizer, args.lr, args.lr_final,
                                  args.max_epochs)
    # Some placeholders.
    intermediate_samples = {0: []}
    train_obj = []
    val_obj = []

    if args.phase == 'train':
        for epoch in range(args.max_epochs):
            # Validation.
            model.eval()
            with torch.no_grad():

                x_val0 = quadratic(n=args.val_batchsize,
                                   s=args.input_size,
                                   x_range=(-3, 3),
                                   eval_pattern='jitter',
                                   phase='test',
                                   device=device)

                noise = torch.randn(x_val0[:, 0, ...].shape, device=device)
                timesteps = torch.randint(0,
                                          len(noise_scheduler),
                                          (x_val0.shape[0], ),
                                          device=device).long()

                x_valt = noise_scheduler.add_noise(x_val0[:, 0, ...], noise,
                                                   timesteps)
                noise_pred = model(x_valt, x_val0[:, 1, ...], timesteps)
                obj = (1 / x_val0.shape[0]) * torch.norm(noise_pred - noise)**2

                val_obj.append(obj.item())

            # Training.
            model.train()
            # Update learning rate.
            scheduler.step()
            with tqdm(train_idx_loader,
                      unit=' itr',
                      colour='#B5F2A9',
                      dynamic_ncols=True) as pb:
                for x in pb:
                    input_size = torch.randint(args.input_size // 2,
                                               args.input_size * 2,
                                               (1, )).item()
                    x0 = quadratic(n=args.val_batchsize,
                                   s=input_size,
                                   x_range=(-3, 3),
                                   eval_pattern='jitter',
                                   phase='train',
                                   device=device)
                    noise = torch.randn(x0[:, 0, ...].shape, device=device)

                    # Randomly sample timesteps.
                    timesteps = torch.randint(0,
                                              len(noise_scheduler),
                                              (x0.shape[0], ),
                                              device=device).long()

                    # Add noise to the data according to the noise schedule.
                    xt = noise_scheduler.add_noise(x0[:, 0, ...], noise,
                                                   timesteps)

                    # Predict the score at this noise level.
                    noise_pred = model(xt, x0[:, 1, ...], timesteps)
                    # from IPython import embed; embed()

                    # Score matching objective.
                    obj = (1 / x0.shape[0]) * torch.norm(noise_pred - noise)**2
                    obj.backward(obj)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    train_obj.append(obj.item())

                    # Progress bar.
                    pb.set_postfix({
                        'epoch': epoch,
                        'train obj': "{:.2f}".format(train_obj[-1]),
                        'val obj': "{:.2f}".format(val_obj[-1])
                    })

            # Save the current network parameters, optimizer state variables,
            # current epoch, and objective log.
            if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_obj': train_obj,
                        'val_obj': val_obj,
                        'args': args
                    },
                    os.path.join(checkpointsdir(args.experiment),
                                 'checkpoint_' + str(epoch) + '.pth'))

    elif args.phase == 'test':

        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            'checkpoint_' + str(args.testing_epoch) + '.pth')

        x_test = quadratic(n=args.testing_nsamples,
                           s=args.input_size,
                           x_range=(-3, 3),
                           eval_pattern='same',
                           phase='test',
                           device=device)

        if os.path.isfile(file_to_load):
            if device == torch.device('cpu'):
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)

            model.load_state_dict(checkpoint['model_state_dict'])
            train_obj = checkpoint["train_obj"]
            val_obj = checkpoint["val_obj"]

            assert args.testing_epoch == checkpoint["epoch"]

        model.eval()
        model.to(device)
        sample_list = []
        with torch.no_grad():
            # Sample intermediate results.
            test_conditioning_input = x_test[0, 1, :].unsqueeze(0).repeat(
                args.val_batchsize, 1)

            timesteps = list(
                torch.arange(len(noise_scheduler),
                             device=device,
                             dtype=torch.int))[::-1]

            # Setup the batch index generator.
            sample_idx_loader = torch.utils.data.DataLoader(
                range(args.testing_nsamples),
                batch_size=args.val_batchsize,
                shuffle=False,
                drop_last=True)

            for idx in sample_idx_loader:
                sample = torch.randn((args.val_batchsize, args.input_size),
                                     device=device)

                for i, t in enumerate(tqdm(timesteps)):
                    t = t.repeat(args.val_batchsize)

                    residual = model(sample, test_conditioning_input, t)
                    sample = noise_scheduler.step(residual, t[0], sample)
                sample_list.append(sample)

            sample_list = torch.cat(sample_list, dim=0).cpu().numpy()

        file = h5py.File(
            os.path.join(checkpointsdir(args.experiment),
                         'collected_samples.h5'), 'a')
        file.require_dataset(str(args.input_size),
                             shape=(args.testing_nsamples, args.input_size),
                             dtype=np.float32)
        file.require_dataset('real_' + str(args.input_size),
                             shape=(args.testing_nsamples, args.input_size),
                             dtype=np.float32)
        file.require_dataset('x_' + str(args.input_size),
                             shape=(args.input_size, ),
                             dtype=np.float32)

        file[str(args.input_size)][...] = sample_list
        file['real_' +
             str(str(args.input_size))][...] = x_test[:, 0, :].cpu().numpy()
        file['x_' + str(args.input_size)][...] = x_test[0, 1, :].cpu().numpy()
        file.close()

        plot_toy_conditional_example_results(args, train_obj, val_obj, x_test,
                                             sample_list)


if '__main__' == __name__:

    args_list = query_experiments(CONFIG_FILE)
    for args in args_list:
        if args.testing_epoch == -1:
            args.testing_epoch = args.max_epochs - 1
        train(args)
