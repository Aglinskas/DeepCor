import argparse
from training_realistic_simulation import train_deepcor, train_compcor
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
    parser.add_argument('--num-length', type=int, default=156)
    parser.add_argument('--percent-obs', type=float, default=1.0)
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--is-linear', type=bool, default=True)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--num-pcs', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    args.savedir = f'{args.savedir}percent-obs_{args.percent_obs}_num-length_{args.num_length}'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # load the dataset
    data_path = '/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/'
    gt_list = np.loadtxt(f'{data_path}gt_list.csv', delimiter=",", dtype=float)
    obs_list = np.loadtxt(f'{data_path}obs_list.csv', delimiter=",", dtype=float)
    noi_list = np.loadtxt(f'{data_path}noi_list.csv', delimiter=",", dtype=float)

    # initiate dataset in pytorch (and crop as needed)
    if args.num_length != 156:
        cropped_gt_list = gt_list[:, 0:args.num_length]
        cropped_obs_list = obs_list[:, 0:args.num_length]
        cropped_noi_list = noi_list[:, 0:args.num_length]
        inputs_all = TrainDataset(cropped_obs_list, cropped_gt_list, cropped_noi_list)
        in_dim = cropped_gt_list.shape[1]
        print("Time course length is "+str(cropped_gt_list.shape[1]))
    else:
        inputs_all = TrainDataset(obs_list, gt_list, noi_list)
        in_dim = gt_list.shape[1]
        print("Time course length is "+str(gt_list.shape[1]))

    # Randomly select certain amounts of samples
    train_num = int(args.percent_obs*0.70*obs_list.shape[0])
    val_num = int(0.15*obs_list.shape[0])
    test_num = int(0.15*obs_list.shape[0])
    total_num_samples = val_num + test_num + train_num
    selected_indices = np.random.choice(len(inputs_all), size=total_num_samples, replace=False)
    selected_samples = torch.utils.data.Subset(inputs_all, selected_indices)

    generator = torch.Generator().manual_seed(args.seed)
    train_inputs, val_inputs, test_inputs = random_split(selected_samples, [train_num, val_num, test_num], generator=generator)
    print("Train number is "+str(len(train_inputs)))
    print("Val number is "+str(len(val_inputs)))
    print("Test number is "+str(len(test_inputs)))

    # dataloading
    train_in = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_in = torch.utils.data.DataLoader(val_inputs, batch_size=len(val_inputs), shuffle=False, num_workers=1)
    test_in = torch.utils.data.DataLoader(test_inputs, batch_size=len(test_inputs), shuffle=False, num_workers=1)
    
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
