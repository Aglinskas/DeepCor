from models import cVAE
from utils import r_squared_list, EarlyStopper, Scaler
import torch
import torch.optim as optim 
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model

def train_deepcor(latent_dim, train_in, val_in, test_in, in_dim):
    model = cVAE(in_channels=1, in_dim=in_dim, latent_dim=latent_dim, hidden_dims=[64, 128, 256, 256])
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    epoch_num = 200
    train_loss_L = []
    train_recons_L = []
    train_KLD_L = []
    val_loss_L = []
    val_recons_L = []
    val_KLD_L = []
    test_correlation_L = []
    test_loss_n_L = []
    test_percentile_L = []
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, epoch_num-1))
        print('-' * 10)

        train_loss = 0.0
        train_reconstruction_loss = 0.0
        train_KLD = 0.0
        val_loss = 0.0
        val_reconstruction_loss = 0.0
        val_KLD = 0.0

        # Iterate over data.
        dataloader_iter_in = iter(train_in)
        for i in range(len(train_in)):
            inputs_gm,inputs_gt,inputs_cf = next(dataloader_iter_in)

            inputs_gm = inputs_gm.unsqueeze(1).float().to(device)
            inputs_cf = inputs_cf.unsqueeze(1).float().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # encoder + decoder
            [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x] = model.forward_tg(inputs_gm)
            [outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s] = model.forward_bg(inputs_cf)
            outputs = torch.concat((outputs_gm,outputs_cf),1)
            loss = model.loss_function(outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x, outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s)
            # backward + optimize
            loss['loss'].backward()
            optimizer.step()
            # print statistics
            train_loss += loss['loss']
            train_reconstruction_loss += loss['Reconstruction_Loss']
            train_KLD += loss['KLD']
        # validation
        with torch.no_grad():
            val_gm, val_gt, val_cf = next(iter(val_in))
            val_gm = val_gm.unsqueeze(1).float().to(device)
            val_cf = val_cf.unsqueeze(1).float().to(device)
            [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x] = model.forward_tg(val_gm)
            [outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s] = model.forward_bg(val_cf)
            loss_val = model.loss_function(outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x, outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s)
            if early_stopper.early_stop(loss_val['loss']):
                break

        epoch_train_loss = train_loss / (len(train_in)*2)
        epoch_train_reconstruction_loss = train_reconstruction_loss / (len(train_in)*2)
        epoch_train_KLD = train_KLD / (len(train_in)*2)
        epoch_val_loss = loss_val['loss']
        epoch_val_reconstruction_loss= loss_val['Reconstruction_Loss']
        epoch_val_KLD = loss_val['KLD']
        print('Training Loss: {:.4f} Training Reconstruction Loss: {:.4f} Training KLD {:.4f}'.format(epoch_train_loss, epoch_train_reconstruction_loss, epoch_train_KLD))
        print('Val Loss: {:.4f} Val Reconstruction Loss: {:.4f} Val KLD {:.4f})'.format(epoch_val_loss,epoch_val_reconstruction_loss,epoch_val_KLD))
        print('')
        print()
        train_loss_L.append(epoch_train_loss)
        train_recons_L.append(epoch_train_reconstruction_loss)
        train_KLD_L.append(epoch_train_KLD)
        val_loss_L.append(epoch_val_loss)
        val_recons_L.append(epoch_val_reconstruction_loss)
        val_KLD_L.append(epoch_val_KLD)

    print('Finished Training')

    test_gm, test_gt, test_cf = next(iter(test_in))
    test_gm = test_gm.unsqueeze(1).float().to(device)
    test_gt = test_gt.unsqueeze(1).float().to(device)
    test_cf = test_cf.unsqueeze(1).float().to(device)
    [output_test, input_test, fg_mu_z, fg_log_var_z] = model.forward_fg(test_gm)
    # loss_test = model.loss_function_val(output_test, input_test, fg_mu_z, fg_log_var_z)
    # print('Test Loss: {:.4f} Test Reconstruction Loss: {:.4f} Test KLD {:.4f})'.format(loss_test['val_loss'],loss_test['val_recons_Loss'],loss_test['val_KLD']))

    #output_test, input_test
    output_scale = Scaler(output_test.squeeze().cpu().detach().numpy())
    outputs_test_n = output_scale.transform(output_test.squeeze().cpu().detach().numpy())
    test_r_squared_list = r_squared_list(test_gt.squeeze().cpu().detach().numpy(),outputs_test_n)
    return model, test_r_squared_list

def train_compcor(num_pcs, train_in, val_in, test_in):
    dataloader_iter_in = iter(train_in)
    train_gm,train_gt,train_cf = next(dataloader_iter_in)
    for i in range(1,len(train_in)):
        train_gm_new,train_gt_new,train_cf_new = next(dataloader_iter_in)
        train_gm = np.concatenate((train_gm,train_gm_new),axis=0)
        train_cf = np.concatenate((train_cf,train_cf_new),axis=0)

    test_gm, test_gt, test_cf = next(iter(test_in))
    test_gm = test_gm.numpy()
    test_gt = test_gt.numpy()
    test_cf = test_cf.numpy()

    # PCA likes the time dimension as first. Let's transpose our data.
    train_gm_t = np.transpose(train_gm)
    train_cf_t = np.transpose(train_cf)
    test_gm_t = np.transpose(test_gm)
    test_cf_t = np.transpose(test_cf)
    # Fit PCA and extract PC timecourses
    pca = PCA(n_components = num_pcs)
    confounds_pc = pca.fit_transform(train_cf_t)
    
    # linear regression on each voxel: PCs -> voxel pattern
    linear = linear_model.LinearRegression()
    linear.fit(confounds_pc, test_gm_t)

    # predict the activity of each voxel for this run
    predict = linear.predict(confounds_pc)
    func_denoised = test_gm_t - predict # t x v
    func_denoised = np.transpose(func_denoised) # v x t
    compcor_r_squared_list = r_squared_list(test_gt,func_denoised)
    return compcor_r_squared_list

