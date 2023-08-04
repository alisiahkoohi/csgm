import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from csgm import NoiseScheduler, ConditionalScoreModel2D
from csgm.utils import (get_conditional_dataset, make_experiment_name,
                        plot_seismic_imaging_results,
                        plot_toy_conditional_example_results, checkpointsdir,
                        plotsdir, query_experiments, CustomLRScheduler,
                        save_exp_to_h5, load_exp_from_h5, make_grid)

CONFIG_FILE = 'seismic_imaging_conditional.json'


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

    # Load the dataset.
    dset_train, dset_val, x_normalizer, y_normalizer = get_conditional_dataset(
        args.dataset,
        n_val=args.val_batchsize,
        input_size=args.input_size,
        device=device)

    grid_train = make_grid(dset_val.tensors[0].shape[2:]).to(device).repeat(
        args.batchsize, 1, 1, 1)
    grid_val = make_grid(dset_val.tensors[0].shape[2:]).to(device).repeat(
        args.val_batchsize, 1, 1, 1)

    # Make the dataloaderf for batching.
    dataloader = DataLoader(dset_train,
                            batch_size=args.batchsize,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=False)

    # Make the dataloaderf for batching.
    val_dataloader = DataLoader(dset_val,
                                batch_size=args.val_batchsize,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)

    # Initialize the network that will learn the score function.
    model = ConditionalScoreModel2D(
        modes=args.modes,
        hidden_dim=args.hidden_dim,
        nlayers=args.nlayers,
        nt=args.nt,
    ).to(device)

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
                var_obj_step = 0.0
                for x_val in val_dataloader:
                    x_val0 = x_val[0].to(device)

                    noise = torch.randn(x_val0[:, 0, ...].shape, device=device)
                    timesteps = torch.randint(0,
                                              len(noise_scheduler),
                                              (x_val0.shape[0], ),
                                              device=device).long()

                    x_valt = noise_scheduler.add_noise(x_val0[:, 0, ...],
                                                       noise, timesteps)
                    noise_pred = model(x_valt, x_val0[:, 1, ...], timesteps,
                                       grid_val)
                    obj = (1 / x_val0.shape[0]) * torch.norm(noise_pred -
                                                             noise)**2
                    var_obj_step += obj.item()

                val_obj.append(var_obj_step / len(val_dataloader))

            # Training.
            model.train()
            # Update learning rate.
            scheduler.step()
            with tqdm(dataloader,
                      unit=' itr',
                      colour='#B5F2A9',
                      dynamic_ncols=True) as pb:
                for x in pb:
                    x0 = x[0].to(device)
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
                    noise_pred = model(xt, x0[:, 1, ...], timesteps,
                                       grid_train)
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

        # from IPython import embed
        # embed()

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
            if args.test_idx == -1:
                args.test_idx = np.random.randint(dset_val.tensors[0].shape[0])
            test_conditioning_input = dset_val.tensors[0][
                args.test_idx, 1, ...].unsqueeze(0).to(device)

            timesteps = list(
                torch.arange(len(noise_scheduler),
                             device=device,
                             dtype=torch.int))[::-1]

            test_conditioning_input = test_conditioning_input.repeat(
                args.val_batchsize, 1, 1, 1)

            # Setup the batch index generator.
            sample_idx_loader = torch.utils.data.DataLoader(
                range(args.testing_nsamples),
                batch_size=args.val_batchsize,
                shuffle=False,
                drop_last=True)

            for idx in sample_idx_loader:
                sample = torch.randn((args.val_batchsize, *args.input_size, 1),
                                     device=device)

                for i, t in enumerate(tqdm(timesteps)):
                    t = t.repeat(args.val_batchsize)

                    residual = model(sample, test_conditioning_input, t,
                                     grid_val)
                    sample = noise_scheduler.step(residual, t[0], sample)
                sample_list.append(sample)

            sample_list = torch.cat(sample_list, dim=0)[..., 0].cpu()

        sample_list = x_normalizer.unnormalize(sample_list.permute(
            0, 2, 1)).permute(0, 2, 1).numpy()

        true_image = x_normalizer.unnormalize(
            dset_val.tensors[0][args.test_idx, 0, ...,
                                0].unsqueeze(0).cpu().permute(
                                    0, 2, 1)).permute(0, 2, 1).numpy()

        rtm_image = y_normalizer.unnormalize(
            dset_val.tensors[0][args.test_idx, 1, ...,
                                0].unsqueeze(0).cpu().permute(
                                    0, 2, 1)).permute(0, 2, 1).numpy()

        # Save the results.
        save_exp_to_h5(os.path.join(plotsdir(args.experiment), 'checkpoint_' +
                                    str(args.test_idx) + '.h5'),
                       args,
                       train_obj=train_obj,
                       val_obj=val_obj,
                       sample_list=sample_list,
                       true_image=true_image,
                       rtm_image=rtm_image)

    elif args.phase == 'plot':
        data_dict = load_exp_from_h5(
            os.path.join(plotsdir(args.experiment),
                         'checkpoint_' + str(args.test_idx) + '.h5'),
            'train_obj', 'val_obj', 'sample_list', 'true_image', 'rtm_image')

        plot_seismic_imaging_results(args, data_dict['train_obj'],
                                     data_dict['val_obj'],
                                     data_dict['sample_list'],
                                     data_dict['true_image'],
                                     data_dict['rtm_image'], args.test_idx)


if '__main__' == __name__:

    args_list = query_experiments(CONFIG_FILE)
    for args in args_list:
        if args.testing_epoch == -1:
            args.testing_epoch = args.max_epochs - 1
        train(args)
