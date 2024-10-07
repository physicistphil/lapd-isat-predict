import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchsummary import summary  # Print model / parameter details

import numpy as np
from tqdm import tqdm
import wandb
import os
import datetime
import time
import shutil
import importlib
import argparse
import json
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani

# Pretty tracebacks
# import rich.traceback
# rich.traceback.install()

import sys
import signal
from multiprocessing import shared_memory as sm


def load_data(path):
    datafile = np.load(path)

    x = datafile['x']
    y = datafile['y']

    return x, y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        datafile = np.load(path)

        x = torch.from_numpy(datafile['x'].astype(np.float32))
        y = torch.from_numpy(datafile['y'].astype(np.float32))

        self.x = x
        self.y = y

        # self.x = torch.rand(*x.shape)[0:16384, 0:2]
        # self.y = self.x[:, 0].detach().clone()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def shape(self):
        return self.x.shape


class ModelClass(torch.nn.Module):
    def __init__(self, hyperparams):
        super(ModelClass, self).__init__()
        # self.act = torch.nn.Tanh()
        self.act = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dense_list = []
        for i in range(hyperparams['num_layers']):
            self.dense_list.append(torch.nn.LazyLinear(hyperparams['layer_width']))
        self.dense_list = torch.nn.ModuleList(self.dense_list)
        self.dense_out = torch.nn.LazyLinear(2)

    def forward(self, x):
        for layer in self.dense_list:
            x = layer(x)
            x = self.act(x)
        x = self.dense_out(x)
        x[:, 1].pow_(2)  # square the stddev to get the variance (to avoid negative variance)
        return x.squeeze()


# A comprehensive guide to distributed data parallel:
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


# Gradient clipping method (based on percentile) ripped from https://github.com/pseeth/autoclip
def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def autoclip_gradient(model, grad_history, clip_percentile, clip_constant_norm):
    obs_grad_norm = _get_grad_norm(model)
    grad_history.append(obs_grad_norm)
    while len(grad_history) > 1000:
        grad_history.pop(0)
    clip_value = np.percentile(grad_history, clip_percentile)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_constant_norm)


def main(rank, world_size, hyperparams, port):
    # Distributed setup
    setup(rank, world_size, port)
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getppid()),)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Handle directories, file copying, etc...
    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    hyperparams['identifier'] = identifier
    project_name = "train_dense_beta_NLL"
    exp_path = "training_runs/"
    ensemble = hyperparams["ensemble"]
    if len(ensemble) > 0:  # If nothing, do not watn a slash
        ensemble += "/"
    path = exp_path + ensemble + identifier
    if rank == 0:
        os.makedirs(path)
        os.makedirs(path + "/checkpoints")
        os.makedirs(path + "/plots")
        shutil.copy(project_name + ".py", path + "/" + project_name + "_copy.py")

        with open(path + "/" + "hyperparams.json", 'w') as json_f:
            json.dump(hyperparams, json_f)
            json_f.write("\n")

    # Set local hyperparameter variables
    num_epochs = hyperparams["num_epochs"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]
    momentum = hyperparams["momentum"]
    beta1 = hyperparams["beta1"]
    beta2 = hyperparams["beta2"]
    weight_decay = hyperparams["weight_decay"]
    clip_percentile = hyperparams["clip_percentile"]
    clip_constant_norm = hyperparams["clip_constant_norm"]

    beta_NLL = hyperparams["beta_NLL"]
    interval_histograms = hyperparams["interval_histograms"]
    interval_checkpoint = hyperparams["interval_checkpoint"]
    interval_scalars = hyperparams["interval_scalars"]
    seed = hyperparams["seed"]

    train_frac = hyperparams["train_frac"]
    dataset_valid = hyperparams["dataset_valid"]
    dataruns_exclude = hyperparams["dataruns_exclude"]
    verify_epochs = hyperparams["verify_epochs"]
    verify_batches = hyperparams["verify_batches"]

    resume = hyperparams["resume"]
    if resume:
        resume_path = hyperparams["resume_path"]
        resume_version = hyperparams["resume_version"]

    # Set seeds for everything
    torch.manual_seed(seed)
    generator1 = torch.Generator().manual_seed(seed)
    np.random.seed(seed)

    # For writing to Tensorboard
    writer = SummaryWriter(log_dir=path)

    # Creating/loading model
    model = ModelClass(hyperparams)
    if resume:
        with open(exp_path + resume_path + "/" + "hyperparams.json") as json_f:
            hyperparams_temp = json.loads(json_f.read())
        spec = importlib.util.spec_from_file_location(project_name + "_copy", exp_path +
                                                      resume_path + "/" + project_name + "_copy.py")
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
        model = loaded_module.ModelClass(hyperparams_temp)

    # Load data
    if rank == 0:
        print("Loading data: " + path, flush=True)
    data_path = "../datasets/" + hyperparams["dataset"]

    data = Dataset(data_path)
    # Exclude dataruns defined on the command line / in hyperparameters
    if len(dataruns_exclude) > 0:
        dr_idx = {**np.load(data_path.replace('isat', 'dr-idx'))}
        datarun_mask = torch.ones(len(data), dtype=torch.bool)
        for dr in dataruns_exclude:
            datarun_mask[dr_idx[dr]] = torch.zeros(len(dr_idx[dr]), dtype=torch.bool)
        data.x = data.x[datarun_mask]
        data.y = data.y[datarun_mask]
        if rank == 0:
            print("Excluding runs: {} shots removed".format(torch.sum(~datarun_mask).item()))

    data_valid = Dataset("../datasets/" + dataset_valid)
    data_valid = torch.utils.data.DataLoader(data_valid, batch_size=len(data_valid), shuffle=False, num_workers=1,
                                             pin_memory=True, persistent_workers=True)
    valid_examples = next(iter(data_valid))
    x_valid = valid_examples[0].to(rank)
    y_valid = valid_examples[1].to(rank)

    # num_examples = data.__len__()
    # data_size = data.shape()
    if rank == 0:
        print("Data shape: ", end="")
        print(data.shape(), flush=True)

    # initialze for lazy layers so that the num_parameters works properly
    model = model.to(rank)
    model(torch.zeros((2, data.shape()[1])).to(rank))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Start sampler, dataloader, and split the data
    data_train, data_test = torch.utils.data.random_split(data,
                                                          [int(np.floor(train_frac * data.__len__())),
                                                           int(np.ceil((1 - train_frac) * data.__len__()))],
                                                          generator=generator1)
    print("Data train: {} \t Data test: {}".format(len(data_train), len(data_test)))

    # train_data = data_list[0]
    # data_test = torch.stack([t for t in data_test]).to(rank)

    train_sampler = DistributedSampler(data_train, num_replicas=world_size,
                                       rank=rank, shuffle=True,
                                       drop_last=False)
    test_sampler = DistributedSampler(data_test, num_replicas=world_size,
                                      rank=rank, shuffle=True,
                                      drop_last=False)
    train_dataloader = DataLoader(data_train,
                                  batch_size=batch_size_max, shuffle=False, num_workers=4,
                                  pin_memory=True, sampler=train_sampler, persistent_workers=True)
    # len(data_test) so that we scan over the entire test set to reduce eval stochasticity
    test_dataloader = DataLoader(data_test,
                                 batch_size=len(data_test), shuffle=False, num_workers=4,
                                 pin_memory=True, sampler=test_sampler, persistent_workers=True)
    # test_iterable = iter(test_dataloader)
    test_examples = next(iter(test_dataloader))
    x_test = test_examples[0].to(rank)
    y_test = test_examples[1].to(rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(beta1, beta2))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
    #                             momentum=momentum, nesterov=False)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay,
    #                               betas=(0.9, 0.999))
    # del data

    # Resume and laod the weights and optimizer state
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt", map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    num_batches = len(train_dataloader)

    # Define learnign rate / warmup schedule for the model and if resuming, load the state
    def lr_func(x):
        # return np.exp(- (x / num_batches / 140))

        return 1 / (1 + x / num_batches / 70)

        # if x / num_batches <= 70 :
        #     return 1.0
        # else:
        #     # return 1 / torch.tensor((x - 500)).to(rank)
        #     return 1 / torch.sqrt(torch.tensor(x / num_batches - 70)).to(rank)
    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    # lrScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    if resume:
        lrScheduler.load_state_dict(ckpt['lrScheduler_state_dict'])

    # Print number of parameters
    if rank == 0:
        # summary(model, (data_size,), batch_size=batch_size_max)
        num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("Parameters: {}".format(num_parameters))
        for name, module in model.named_modules():
            print(name, sum(param.numel() for param in module.parameters()))
        hyperparams['num_parameters'] = num_parameters
    # Initialize weights and biases
        wandb.init(project="profile-predict", entity='phil',
                   group="PP1", job_type="",
                   config=hyperparams)

    # print(test_data.shape)

    t_start0 = t_start1 = t_start2 = t_start_autoclose = time.time()
    test_loss_best = 1e5
    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    # grad_mag_list = []
    if resume:
        batch_iteration = ckpt['batch_iteration']
    # model.train(True)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    list_loss_train = []
    list_loss_test = []
    list_loss_valid = []
    list_mse_train = []
    list_mse_test = []
    list_mse_valid = []
    grad_history = []  # Storing gradients for percentile-based clipping
    Gauss_NLL_loss = torch.nn.GaussianNLLLoss(reduction='none')  # defaults are ok

    def loss_func(output, y_feed, beta_NLL):
        with torch.no_grad():
            beta_factor = torch.pow(output[:, 1], beta_NLL)
        loss = Gauss_NLL_loss(output[:, 0], y_feed[:], output[:, 1]) * beta_factor
        return torch.mean(loss)

    # For verifying that data is being fed into the model correctly
    if len(verify_batches) > 0:
        if len(verify_epochs) == 0:
            verify_epochs = [0]
        data_verify_x = []
        data_verify_y = []
    if len(verify_epochs) > 0 and len(verify_batches) == 0:
        verify_batches = list(range(len(num_batches)))
        data_verify_x = []
        data_verify_y = []

    for epoch in range(num_epochs):
        batch_pbar = tqdm(total=num_batches)
        # for (x_feed, y_feed), i in zip(train_dataloader, range(num_batches)):
        for i, data_feed in enumerate(train_dataloader):
            # if i == 1:  # If overfitting to one batch only
            #     break

            x_feed = data_feed[0].to(rank)
            y_feed = data_feed[1].to(rank)

            if epoch in verify_epochs and i in verify_batches:
                data_verify_x.append(x_feed.clone().detach().cpu().numpy())
                data_verify_y.append(y_feed.clone().detach().cpu().numpy())

            # with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast(enabled=False):

                # Backwards pass...
                optimizer.zero_grad()
                output = model(x_feed)

                # MSE loss
                # loss = torch.nn.MSELoss(output, y)  # reduction = mean by default
                # loss = torch.mean((output - y_feed) ** 2)

                # Negative log-likelihood loss for regression assuming a Gaussian distribution
                loss = loss_func(output, y_feed, beta_NLL)
                mse = torch.mean((model(x_feed)[:, 0] - y_feed) ** 2)

            # loss.backward()
            scaler.scale(loss).backward()

            # For debugging missing gradients error
            # for name, p in model.named_parameters():
            #     if p.grad is None:
            #         print("found unused param: ")
            #         print(name)
            #         print(p)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            autoclip_gradient(model, grad_history, clip_percentile, clip_constant_norm)
            scaler.step(optimizer)
            scaler.update()
            lrScheduler.step()

            # print(linearScheduler.get_last_lr())
            # print(sqrtScheduler.get_last_lr())
            loss = loss.detach().cpu()
            mse = mse.detach().cpu()
            if rank == 0:
                # print("\n")
                # print(reg_avg)

                # rmse = torch.sqrt(loss)
                list_loss_train.append((epoch, i, batch_iteration, loss.numpy()))
                list_mse_train.append((epoch, i, batch_iteration, mse.numpy()))

                if i % 20 == 0:
                    # try:
                    #     test_examples = next(test_iterable)
                    # except StopIteration:
                    #     test_iterable = iter(test_dataloader)
                    #     test_examples = next(test_iterable)
                    # test_examples = data_test
                    output_test = model(x_test)
                    # loss_test = torch.mean((model(x_test) - y_test) ** 2)
                    loss_test = loss_func(output_test, y_test, beta_NLL).detach().cpu().numpy()
                    mse_test = torch.mean((model(x_test)[:, 0] - y_test) ** 2).detach().cpu().numpy()
                    list_loss_test.append((epoch, i, batch_iteration, loss_test))
                    list_mse_test.append((epoch, i, batch_iteration, mse_test))

                    output_valid = model(x_valid)
                    # loss_valid = torch.mean((model(x_valid) - y_valid) ** 2)
                    loss_valid = loss_func(output_valid, y_valid, beta_NLL).detach().cpu().numpy()
                    mse_valid = torch.mean((model(x_valid)[:, 0] - y_valid) ** 2).detach().cpu().numpy()
                    list_loss_valid.append((epoch, i, batch_iteration, loss_valid))
                    list_mse_valid.append((epoch, i, batch_iteration, mse_valid))

                    tqdm.write("#: {} // L: {:.2e} // L_test: {:.2e} // L_valid: {:.2e} // MSE: {:.2e}".format(i, loss, loss_test, loss_valid, mse))
                    wandb.log({"loss/train": loss,
                               "loss/test": loss_test,
                               "loss/valid": loss_valid,
                               "loss/mse_train": mse,
                               "loss/mse_test": mse_test,
                               "loss/mse_valid": mse_valid,
                               "batch_num": batch_iteration,
                               "epoch": epoch})
                else:
                    tqdm.write("#: {} // L: {:.2e} // RMSE: {:.2e}".format(i, loss, mse), end='\r')
                    wandb.log({"loss/train": loss,
                               "loss/mse_train": mse,
                               "batch_num": batch_iteration,
                               "epoch": epoch})

            # Longer-term metrics
            if rank == 0:
                # End training after fixed amount of time
                if time.time() - t_start_autoclose > 3600 * hyperparams['time_limit'] and hyperparams['time_limit'] > 0:
                    sh_mem.buf[0] = 1

                # Log to tensorboard every so often
                if (epoch == 0 and i == 3) or time.time() - t_start0 > interval_scalars or sh_mem.buf[0] == 1:
                    t_start0 = time.time()
                    # write scalars and histograms
                    writer.add_scalar("loss/train", loss.detach().cpu(), batch_iteration)  # used to be loss_avg
                    writer.add_scalar("loss/test", list_loss_test[-1][3], list_loss_test[-1][2])
                    writer.add_scalar("loss/valid", list_loss_valid[-1][3], list_loss_valid[-1][2])
                    writer.add_scalar("loss/mse_train", mse.detach().cpu(), batch_iteration)  # used to be loss_avg
                    writer.add_scalar("loss/mse_test", list_mse_test[-1][3], list_mse_test[-1][2])
                    writer.add_scalar("loss/mse_valid", list_mse_valid[-1][3], list_mse_valid[-1][2])

                # Add histogram min
                if (epoch == 0 and i == 3) or time.time() - t_start1 > interval_histograms or sh_mem.buf[0] == 1:
                    t_start1 = time.time()
                    try:
                        for name, weight in model.named_parameters():
                            writer.add_histogram("w/" + name, weight, batch_iteration)
                            writer.add_histogram(f'g/{name}.grad', weight.grad, batch_iteration)
                    except Exception as e:
                        print(e)
                    writer.flush()

                                # Save checkpoint occasionally, or when test lost is the best so far
                if (loss_test < test_loss_best and epoch >= 1):
                    test_loss_best = loss_test
                    model_save_path = path + "/checkpoints/model-best.pt"
                    torch.save({'epoch': epoch,
                                'batch_iteration': batch_iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lrScheduler_state_dict': lrScheduler.state_dict(),
                                }, model_save_path)
                if ((epoch == 0 and i == 3) or (epoch == num_epochs - 1 and i == num_batches - 1) or
                    time.time() - t_start2 > interval_checkpoint or sh_mem.buf[0] == 1):
                    model_save_path = path + "/checkpoints/model-{}-{}.pt".format(epoch, i)
                    t_start2 = time.time()
                    torch.save({'epoch': epoch,
                                'batch_iteration': batch_iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lrScheduler_state_dict': lrScheduler.state_dict(),
                                }, model_save_path)


            batch_iteration += 1
            batch_pbar.update(1)

            if sh_mem.buf[0] == 1:
                print("\n ---- Exiting process ----\n")
                break

        # lrScheduler.step()

        if len(verify_batches) > 0 and epoch in verify_epochs:
            np.savez(path + "/batch_data_epoch-{}".format(epoch), x=np.array(data_verify_x, dtype=np.float32),
                     y=np.array(data_verify_y, dtype=np.float32))
            data_verify_x = []
            data_verify_y = []

        batch_pbar.close()
        pbar.update(1)
        if sh_mem.buf[0] == 1:
            break
    pbar.close()
    sh_mem.close()
    if rank == 0:
        wandb.finish()
    cleanup()
    np.savez(path + "/losses", train=np.array(list_loss_train, dtype=np.float32),
             test=np.array(list_loss_test, dtype=np.float32),
             valid=np.array(list_loss_valid, dtype=np.float32),
             mse_train=np.array(list_mse_train, dtype=np.float32),
             mse_test=np.array(list_mse_test, dtype=np.float32),
             mse_valid=np.array(list_mse_valid, dtype=np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modular EBM")
    parser.add_argument('--num_epochs', type=int)

    parser.add_argument('--batch_size_max', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--clip_precentile', type=float)
    parser.add_argument('--clip_constant_norm', type=float)
    parser.add_argument('--beta_NLL', type=float)

    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--layer_width', type=int)
    # parser.add_argument('--identifier')

    parser.add_argument('--ensemble', type=str)
    parser.add_argument('--resume', type=bool)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_version', type=str)

    parser.add_argument('--interval_histograms', type=float)
    parser.add_argument('--interval_checkpoint', type=float)
    parser.add_argument('--interval_scalars', type=float)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_frac', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_valid', type=str)
    parser.add_argument('--dataruns_exclude', nargs='+', type=str)

    parser.add_argument('--verify_epochs', nargs='+', type=int)
    parser.add_argument('--verify_batches', nargs='+', type=int)

    parser.add_argument('--time_limit', type=float, default=-1,
                        help='Time limit (in hours). -1 for unlimited')
    parser.add_argument('--port', type=int, default=26000)
    args = parser.parse_args()

    hyperparams = {
        "num_epochs": 500,

        "batch_size_max": 128,
        "lr": 3e-4,
        "momentum": 0.5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0e-5,
        "clip_percentile": 0.9,
        "clip_constant_norm": 1e4,
        "beta_NLL": 0.5,

        "num_layers": 4,
        "layer_width": 512,

        # "identifier": identifier,
        'ensemble': "",
        "resume": False,
        "resume_path": None,
        "resume_version": None,

        "interval_histograms": 60,
        "interval_checkpoint": 600,
        "interval_scalars": 2,

        'time_limit': -1,

        'seed': 42,
        'train_frac': 0.8,
        'dataset': "DR_combo_PP1_isat_04_train_cv-0.npz",
        'dataset_valid': "DR_combo_PP1_isat_04_valid_cv-0.npz",
        'dataruns_exclude': [],

        'verify_epochs': [],
        'verify_batches': [],
    }

    for key in vars(args).keys():
        if vars(args)[key] is not None:
            hyperparams[key] = vars(args)[key]

    world_size = 1
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getpid()), create=True, size=1)
    sh_mem.buf[0] = 0
    try:
        proc_context = torch.multiprocessing.spawn(main, args=(world_size, hyperparams, args.port),
                                                   nprocs=world_size, join=False)
        proc_context.join()
    except KeyboardInterrupt:
        sh_mem.buf[0] = 1
        proc_context.join(timeout=30)
        sh_mem.unlink()
    else:
        sh_mem.unlink()
