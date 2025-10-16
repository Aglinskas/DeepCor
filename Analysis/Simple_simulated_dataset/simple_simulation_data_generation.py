import argparse
import numpy as np
import os
import math
import random
from numpy import random
from numpy import savetxt

class Scaler():
    ## used to normalize sequences
    def __init__(self,inputs):
        self.data = inputs
        self.mean = np.mean(inputs,axis=1)
        self.std = np.std(inputs, axis=1)
        self.vox, self.time = inputs.shape
    def transform(self,inputs):
        self.mean = np.reshape(self.mean,(self.vox,1))
        self.m_large = np.repeat(self.mean,self.time,axis=1)
        self.std = np.reshape(self.std,(self.vox,1))
        self.s_large = np.repeat(self.std,self.time,axis=1)
        return np.divide(inputs-self.m_large,self.s_large)
    def inverse_transform(self,outputs):
        return np.multiply(outputs,self.s_large)+self.m_large
    

def create_signal(length, is_signal):
    ### functions to generate noise and signal separately as a sine function with noise, and noise and signal has different range of b's
    a_n = random.uniform(5,10)
    b_n = random.uniform(0.1,1)
    c_n = random.uniform(200,300)
    y_n = np.array([a_n*np.sin(i*b_n)+c_n+0.01*random.randint(-100,100) for i in range(length)])
    if is_signal:
        a_s = random.uniform(5,10)
        b_s = random.uniform(1,3)
        c_s = random.uniform(200,300)
        y_s = np.array([a_s*np.sin(i*b_s)+c_s+0.01*random.randint(-100,100) for i in range(length)])
        return [y_n, y_s]
    else:
        return y_n

def generate_dataset(num_obs, length, linear, std):
    obs_gt_list = np.zeros([num_obs, length]) # initionalization of ground truths for observations
    obs_noi_list = np.zeros([num_obs, length]) # initionalization of noise for observations
    noise_list = np.zeros([num_obs, length]) # initionalization of noises
    # generation of noises and ground truths
    for i in range(num_obs):
        [obs_noi_list[i],obs_gt_list[i]] = create_signal(length, True)
        noise_list[i] = create_signal(length, False)

    # subtracting means from noises and ground truths across each seqeunce
    noise_list = noise_list - np.repeat(np.reshape(np.mean(noise_list,axis=1),(noise_list.shape[0],1)),noise_list.shape[1],axis=1)
    obs_gt_list = obs_gt_list - np.repeat(np.reshape(np.mean(obs_gt_list,axis=1),(obs_gt_list.shape[0],1)),obs_gt_list.shape[1],axis=1)
    obs_noi_list = obs_noi_list - np.repeat(np.reshape(np.mean(obs_noi_list,axis=1),(obs_noi_list.shape[0],1)),obs_noi_list.shape[1],axis=1)

    # normalize the grounth truth
    gt_scale = Scaler(obs_gt_list)
    gt_list = gt_scale.transform(obs_gt_list)

    if linear:
        # linear combination of noise and ground truth, leading to the observations, and standard deviation is arbitrary
        lamb = math.sqrt(std**2/obs_noi_list.std()**2)
        noise_c = lamb * obs_noi_list
        observation_list = gt_list+noise_c 
    else:
        # non-linear combination of noise and ground truth, leading to the observations, and standard deviation is arbitrary
        cube_root = np.cbrt(obs_noi_list)
        lamb = (std**2/(cube_root.std())**2)**(3/2)
        noise_c = np.cbrt(lamb * obs_noi_list)
        observation_list = gt_list+noise_c
    # normalization of observations and noises
    obs_scale = Scaler(observation_list)
    obs_list = obs_scale.transform(observation_list)
    noi_scale = Scaler(noise_list)
    noi_list = noi_scale.transform(noise_list)
    print('observation mean is ' + str(obs_list.mean()))
    print('observation std is ' + str(obs_list.std()))
    print('noise mean is ' + str(noi_list.mean()))
    print('noise std is ' + str(noi_list.std()))
     
    return gt_list, obs_list, noi_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--percent-length', type=float, default=1.0)
    parser.add_argument('--percent-obs', type=float, default=1.0)
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--is_linear', type=bool, default=False)
    parser.add_argument('--std', type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    num_obs = int(3000 + 7000 * args.percent_obs) #number of observations
    length = int(156 * args.percent_length)
    print(args.is_linear)
    print(args.std)

    if args.is_linear:
        args.savedir = str(args.savedir) + f'linear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'
    else:
        args.savedir = str(args.savedir) + f'nonlinear_std={args.std}/plen{args.percent_length}_pobs{args.percent_obs}/'

    print(args.savedir)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    gt_list, obs_list, noi_list = generate_dataset(num_obs, length, args.is_linear, args.std)
    savetxt(f'{args.savedir}gt_list.csv', gt_list, delimiter=',')
    savetxt(f'{args.savedir}obs_list.csv', obs_list, delimiter=',')
    savetxt(f'{args.savedir}noi_list.csv', noi_list, delimiter=',')



if __name__ == '__main__':
    main()
