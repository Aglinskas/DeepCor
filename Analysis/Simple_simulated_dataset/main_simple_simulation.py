import argparse
from training_simple_simulation import train_deepcor, train_compcor
from utils import TrainDataset
import torch
import numpy as np
import random
import os
from torch.utils.data import random_split
from numpy import savetxt

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--percent-length', type=float, default=1.0)
    parser.add_argument('--percent-obs', type=float, default=1.0)
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--is-linear', type=bool, default=True)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--num-pcs', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.is_linear:
        args.savedir = str(args.savedir) + f'linear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'
    else:
        args.savedir = str(args.savedir) + f'nonlinear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # load the dataset
    if args.is_linear:
        data_path = '/mmfs1/data/zhupu/Revision/datasets/simple_simulation/' + f'linear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'
    else:
        data_path = '/mmfs1/data/zhupu/Revision/datasets/simple_simulation/' + f'nonlinear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'
    gt_list = np.loadtxt(f'{data_path}gt_list.csv', delimiter=",", dtype=float)
    obs_list = np.loadtxt(f'{data_path}obs_list.csv', delimiter=",", dtype=float)
    noi_list = np.loadtxt(f'{data_path}noi_list.csv', delimiter=",", dtype=float)


    # initiate dataset in pytorch
    inputs_all = TrainDataset(obs_list,gt_list,noi_list)
    generator = torch.Generator().manual_seed(args.seed)
    # train_p = 0.7*args.percent_obs/(0.7*args.percent_obs+0.3)
    # val_p = (1.0-train_p)/2
    train_num = int(7000*args.percent_obs)
    train_inputs, val_inputs, test_inputs = random_split(inputs_all, [train_num, 1500, 1500], generator=generator)
    print("Train length is "+str(len(train_inputs)))
    print("Val length is "+str(len(val_inputs)))
    print("Test length is "+str(len(test_inputs)))

    # dataloading
    train_in = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_in = torch.utils.data.DataLoader(val_inputs, batch_size=len(val_inputs), shuffle=False, num_workers=1)
    test_in = torch.utils.data.DataLoader(test_inputs, batch_size=len(test_inputs), shuffle=False, num_workers=1)
    in_dim = obs_list.shape[1]
    
    model, test_r_squared_list = train_deepcor(args.latent_dim, train_in, val_in, test_in, in_dim)

    test_mean = test_r_squared_list.mean()
    test_percentile= (np.percentile(test_r_squared_list,5), np.percentile(test_r_squared_list,95))
    # save model and loss
    torch.save(model.state_dict(), f'{args.savedir}deepcor_model')
    print("DeepCor testing R squared mean is "+str(test_mean))
    print("DeepCor testing R squared percentile is "+str(test_percentile))
    savetxt(f'{args.savedir}deepcor_r_squared_list.csv', test_r_squared_list, delimiter=',')
    
    compcor_r_squared_list = train_compcor(args.num_pcs, train_in, val_in, test_in)
    compcor_test_mean = compcor_r_squared_list.mean()
    compcor_test_percentile= (np.percentile(compcor_r_squared_list,5), np.percentile(compcor_r_squared_list,95))
    print("CompCor testing R squared mean is "+str(compcor_test_mean))
    print("CompCor testing R squared percentile is "+str(compcor_test_percentile))
    savetxt(f'{args.savedir}compcor_r_squared_list.csv', compcor_r_squared_list, delimiter=',')  

if __name__ == '__main__':
    main()