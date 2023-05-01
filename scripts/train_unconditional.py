import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from csgm import NoiseScheduler, ScoreGenerativeModel
from csgm.utils import (get_dataset, configsdir, read_config, parse_input_args,
                        make_experiment_name, plot_toy_example_results,
                        checkpointsdir, query_experiments, CustomLRScheduler,
                        upload_results)

CONFIG_FILE = 'toy_example.json'


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
    dset_train, dset_val = get_dataset(args.dataset,
                                       n=args.num_train,
                                       n_val=args.val_batchsize,
                                       input_size=args.input_size,
                                       device=device)
    dset_val = dset_val.tensors[0]

    # Make the dataloaderf for batching.
    dataloader = DataLoader(dset_train,
                            batch_size=args.batchsize,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=False)

    # Initialize the network that will learn the score function.
    model = ScoreGenerativeModel(input_size=args.input_size,
                                 hidden_dim=args.hidden_dim,
                                 nlayers=args.nlayers,
                                 emb_size=args.emb_dim,
                                 time_emb=args.time_emb,
                                 input_emb=args.input_emb,
                                 model=args.model).to(device)

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
    intermediate_samples = []
    train_obj = []
    val_obj = []

    if args.phase == 'train':
        with tqdm(range(args.max_epochs),
                  unit='epoch',
                  colour='#B5F2A9',
                  dynamic_ncols=True) as pb:
            for epoch in pb:
                model.train()
                # Update learning rate.
                scheduler.step()
                for x in (dataloader):
                    x0 = x[0]
                    noise = torch.randn(x0.shape, device=device)

                    # Randomly sample timesteps.
                    timesteps = torch.randint(0,
                                              len(noise_scheduler),
                                              (x0.shape[0], ),
                                              device=device).long()

                    # Add noise to the data according to the noise schedule.
                    xt = noise_scheduler.add_noise(x0, noise, timesteps)

                    # Predict the score at this noise level.
                    noise_pred = model(xt, timesteps)

                    # Score matching objective.
                    obj = (1 / x0.shape[0]) * torch.norm(noise_pred - noise)**2
                    obj.backward(obj)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    train_obj.append(obj.item())

                # Validation.
                model.eval()
                with torch.no_grad():
                    timesteps = torch.randint(0,
                                              len(noise_scheduler),
                                              (dset_val.shape[0], ),
                                              device=device).long()
                    noise = torch.randn(dset_val.shape, device=device)
                    xt = noise_scheduler.add_noise(dset_val, noise, timesteps)
                    noise_pred = model(xt, timesteps)
                    obj = (1 / dset_val.shape[0]) * torch.norm(noise_pred -
                                                               noise)**2
                    val_obj.append(obj.item())

                # Progress bar.
                pb.set_postfix({
                    'train obj':
                    "{:.4f}".format(np.mean(train_obj[-len(dataloader):])),
                    'val obj':
                    "{:.4f}".format(val_obj[-1])
                })

                if (epoch % args.save_freq == 0
                        or epoch == args.max_epochs - 1):
                    # Sample intermediate results.
                    timesteps = list(
                        torch.arange(len(noise_scheduler),
                                     device=device,
                                     dtype=torch.int))[::-1]
                    sample = torch.randn(args.val_batchsize,
                                         args.input_size,
                                         device=device)
                    for i, t in enumerate(tqdm(timesteps)):
                        t = t.repeat(args.val_batchsize)
                        with torch.no_grad():
                            residual = model(sample, t)
                            sample = noise_scheduler.step(
                                residual, t[0], sample)
                    intermediate_samples.append(sample.cpu().numpy())

        torch.save(model.state_dict(),
                   os.path.join(checkpointsdir(args.experiment), "model.pth"))

    elif args.phase == 'test':
        model.load_state_dict(
            torch.load(
                os.path.join(checkpointsdir(args.experiment), "model.pth")))
        with torch.no_grad():
            sample = euler_maruyama_solver(noise_scheduler,
                                           model,
                                           args.batchsize,
                                           dset_val.shape[-1],
                                           device=device)
        intermediate_samples.append(sample.cpu().numpy())

    plot_toy_example_results(args, train_obj, val_obj, dset_val,
                             intermediate_samples, noise_scheduler)


if '__main__' == __name__:

    args_list = query_experiments(CONFIG_FILE)
    for args in args_list:
        train(args)

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress')