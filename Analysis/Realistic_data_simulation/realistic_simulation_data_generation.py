
# import packages
from pathlib import Path
from brainiak.utils import fmrisim
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.spatial.distance as sp_distance
import sklearn.manifold as manifold
import scipy.stats as stats
import sklearn.model_selection
import sklearn.svm

nii = nibabel.load('/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/Participant_01_rest_run01.nii')
volume = nii.get_fdata()

dim = volume.shape  # What is the size of the volume
dimsize = nii.header.get_zooms()  # Get voxel dimensions from the nifti header
tr = dimsize[3]
if tr > 100:  # If high then these values are likely in ms and so fix it
    tr /= 1000
print('Volume dimensions:', dim)
print('TR duration: %0.2fs' % tr)
print(dimsize)


mask, template = fmrisim.mask_brain(volume=volume,
                                    mask_self=True,
                                    )


import nibabel as nib
# load functional data
filepath_func = '/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/sub-01_ses-localizer_task-objectcategories_run-1_bold_space-T1w_preproc.nii'
func = nib.load(filepath_func)

# load gray matter
filepath_gm = '/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/sub-01_T1w_class-GM_probtissue.nii'
gm = nib.load(filepath_gm)

#import nibabel.affines
from nibabel.affines import rescale_affine
import nibabel.processing as nibp
# resize gray matter
func0 = func.slicer[:,:,:,0]
# print(func0)
# print(func0.shape)
# print(func0.affine)
gm_funcSize=nibp.resample_from_to(gm, func0, order=1)

# discretize gray matter
gm_values = gm_funcSize.get_fdata()
gm_mask = (gm_values>0.1)


# load white matter and csf
filepath_wm = '/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/sub-01_T1w_class-WM_probtissue.nii'
wm = nib.load(filepath_wm)
# print(wm.shape)
# print(wm.affine)
filepath_csf = '/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/sub-01_T1w_class-CSF_probtissue.nii'
csf = nib.load(filepath_csf)
# print(wm.shape)
# print(csf.affine)

# resize white matter and csf
wm_funcSize=nibp.resample_from_to(wm, func0, order=1)
csf_funcSize=nibp.resample_from_to(csf, func0, order=1)

# discretize white matter and csf and make confounds mask
wm_values = wm_funcSize.get_fdata()
csf_values = csf_funcSize.get_fdata()
confounds_values = wm_values+csf_values
confounds_mask = (confounds_values>0.1)
template_all = gm_values + wm_values + csf_values

# brain_mask = gm_mask | confounds_mask
diff = gm_mask & confounds_mask
gm_mask_c = gm_mask ^ diff
confounds_mask_c = confounds_mask ^diff
brain_mask = gm_mask_c | confounds_mask_c

## generate noise
# Calculate the noise parameters from the data. Set it up to be matched.
noise_dict = {'voxel_size': [dimsize[0],dimsize[1],dimsize[2]], 'matched': 1}
noise_dict = fmrisim.calc_noise(volume=volume,
                                mask=mask,
                                template=template,
                                noise_dict=noise_dict,
                                )


print('Noise parameters of the data were estimated as follows:')
print('SNR: ' + str(noise_dict['snr']))
print('SFNR: ' + str(noise_dict['sfnr']))
print('FWHM: ' + str(noise_dict['fwhm']))


volume_c = (func.shape)
dimsize_c = func.header.get_zooms()
brain_mask_c = 1*brain_mask

# Calculate the noise given the parameters
noise = fmrisim.generate_noise(dimensions=volume_c[0:3],
                               tr_duration=int(dimsize_c[3]),
                               stimfunction_tr=[0] * volume_c[3],
                               mask=brain_mask,
                               template=template_all,
                               noise_dict=noise_dict,
                               )

# # Compute the noise parameters for the simulated noise
# noise_dict_sim = {'voxel_size': [dimsize[0], dimsize[1], dimsize[2]], 'matched': 1}
# noise_dict_sim = fmrisim.calc_noise(volume=noise,
#                                     mask=brain_mask,
#                                     template=template,
#                                     noise_dict=noise_dict_sim,
#                                     )
                                    
# print('Compare noise parameters for the real and simulated noise:')
# print('SNR: %0.2f vs %0.2f' % (noise_dict['snr'], noise_dict_sim['snr']))
# print('SFNR: %0.2f vs %0.2f' % (noise_dict['sfnr'], noise_dict_sim['sfnr']))
# print('FWHM: %0.2f vs %0.2f' % (noise_dict['fwhm'], noise_dict_sim['fwhm']))
# print('AR: %0.2f vs %0.2f' % (noise_dict['auto_reg_rho'][0], noise_dict_sim['auto_reg_rho'][0]))

noise_reshaped = np.reshape(noise,[noise.shape[0]*noise.shape[1]*noise.shape[2],noise.shape[3]])
volume_reshaped = np.reshape(volume,[volume.shape[0]*volume.shape[1]*volume.shape[2],volume.shape[3]])

## generate signal
import itertools
x = [i for i in range(noise.shape[0])]
y = [i for i in range(noise.shape[1])]
z = [i for i in range(noise.shape[2])]
coordinates = list(itertools.product(x,y,z))
print(len(coordinates))

# Create the region of activity where signal will appear
coordinates = np.array(coordinates)  # Where in the brain is the signal
feature_size = 1  # How big, in voxels, is the size of the ROI
signal_volume = fmrisim.generate_signal(dimensions=noise.shape[0:3],
                                        feature_type=['cube'],
                                        feature_coordinates=coordinates,
                                        feature_size=[feature_size],
                                        signal_magnitude=[1],
                                        )


signal_volume_c = np.copy(signal_volume)
signal_volume_c[:,:,0] = 1.0
signal_volume_c[:,0,:] = 1.0
signal_volume_c[0,:,:] = 1.0


# # Create a pattern for each voxel in our signal ROI
# # voxels = feature_size ** 3
voxels = np.count_nonzero(signal_volume_c)
# # print(voxels)

# Pull the conical voxel activity from a uniform distribution
pattern_A = np.random.rand(voxels).reshape((voxels, 1))
pattern_B = np.random.rand(voxels).reshape((voxels, 1))


# Set up stimulus event time course parameters
event_duration = 2  # How long is each event
isi = 7  # What is the time between each event
burn_in = 1  # How long before the first event

total_time = int(noise.shape[3] * dimsize_c[3]) + burn_in  # How long is the total event time course
events = int((total_time - ((event_duration + isi) * 2))  / ((event_duration + isi) * 2)) * 2  # How many events are there?
onsets_all = np.linspace(burn_in, events * (event_duration + isi), events)  # Space the events out
np.random.shuffle(onsets_all)  # Shuffle their order
onsets_A = onsets_all[:int(events / 2)]  # Assign the first half of shuffled events to condition A
onsets_B = onsets_all[int(events / 2):]  # Assign the second half of shuffled events to condition B
temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated?


# Create a time course of events
stimfunc_A = fmrisim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )

stimfunc_B = fmrisim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )

fmrisim.export_epoch_file(stimfunction=[np.hstack((stimfunc_A, stimfunc_B))],
                          filename='/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/epoch_file.npy',
                          tr_duration=tr,
                          temporal_resolution=temporal_res,
                          )

fmrisim.export_3_column(stimfunction=stimfunc_A,
                        filename='/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/Condition_A.txt',
                        temporal_resolution=temporal_res,
                        )

fmrisim.export_3_column(stimfunction=stimfunc_B,
                        filename='/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/Condition_B.txt',
                        temporal_resolution=temporal_res,
                        )

# Multiply each pattern by each voxel time course
weights_A = np.matlib.repmat(stimfunc_A, 1, voxels).transpose() * pattern_A
weights_B = np.matlib.repmat(stimfunc_B, 1, voxels).transpose() * pattern_B

# Sum these time courses together
stimfunc_weighted = weights_A + weights_B
stimfunc_weighted = stimfunc_weighted.transpose()

signal_func = fmrisim.convolve_hrf(stimfunction=stimfunc_weighted,
                                   tr_duration=dimsize_c[3],
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )

# Specify the parameters for signal
signal_method = 'CNR_Amp/Noise-SD'

# Where in the brain are there stimulus evoked voxels
signal_idxs = np.where(signal_volume_c == 1)
print(signal_idxs[0].shape)

snr_list = [0.1, 0.2, 0.5, 1.0]
for snr in snr_list:
    signal_magnitude = [snr]

    signal_func_cp = signal_func.copy()
    signal_volume_cp = signal_volume_c.copy()

    # Pull out the voxels corresponding to the noise volume
    noise_func = noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :].T
    

    # Compute the signal appropriate scaled
    signal_func_scaled = fmrisim.compute_signal_change(signal_func_cp,
                                                    noise_func,
                                                    noise_dict,
                                                    magnitude=signal_magnitude,
                                                    method=signal_method,
                                                    )

    signal = fmrisim.apply_signal(signal_func_scaled,
                                signal_volume_cp,
                                )

    timepoints = 156
    signal_reshaped = np.reshape(signal,[signal.shape[0]*signal.shape[1]*signal.shape[2], signal.shape[3]])
    noise_reshaped = np.reshape(noise,[noise.shape[0]*noise.shape[1]*noise.shape[2],noise.shape[3]])
    gm_reshaped = np.reshape(gm_mask_c,-1)
    confounds_mask_reshaped = np.reshape(confounds_mask_c,-1)
    brain_mask_reshaped = np.reshape(brain_mask,-1)


    overlap = np.logical_and(gm_reshaped, confounds_mask_reshaped)
    are_non_overlapping = not np.any(overlap)
    union = np.logical_or(gm_reshaped, confounds_mask_reshaped)
    is_union = np.array_equal(union, brain_mask_reshaped)
    print("Are gm_reshaped and confounds_mask_c non-overlapping?", are_non_overlapping)
    print("Is brain_mask_reshaped the union of gm_reshaped and confounds_mask_c?", is_union)

    ground_truth_list = signal_reshaped[gm_reshaped,:]
    selected_noise = noise_reshaped[gm_reshaped,:] 
    observation_list = ground_truth_list + selected_noise
    noise_list = noise_reshaped[confounds_mask_reshaped,:]
    print(ground_truth_list.shape)
    print(observation_list.shape)
    print(noise_list.shape)
    np.savetxt(f'/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/simulated_selected_noise_snr{snr}.csv', selected_noise, delimiter=',', fmt='%.6f')
    np.savetxt(f'/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/simulated_obs_list_snr{snr}.csv', observation_list, delimiter=',', fmt='%.6f')
    np.savetxt(f'/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/simulated_gt_list_snr{snr}.csv', ground_truth_list, delimiter=',', fmt='%.6f')
    np.savetxt(f'/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/simulated_noi_list_snr{snr}.csv', noise_list, delimiter=',', fmt='%.6f')

    # observation_volume = np.zeros((func0.shape[0], func0.shape[1], func0.shape[2], timepoints))
    # noise_volume = np.zeros((func0.shape[0], func0.shape[1], func0.shape[2], timepoints))
    # observation_volume[gm_mask_c, :] = observation_list
    # noise_volume[confounds_mask_c,:] = noise_list
    # simulated_fmri_volume = observation_volume + noise_volume
    # simulated_fmri_nii = nibabel.Nifti1Image(simulated_fmri_volume, affine=func.affine)
    # simulated_fmri_nii.header['pixdim'][4] = tr
    # nibabel.save(simulated_fmri_nii, f'/mmfs1/data/zhupu/Revision/datasets/realistic_simulation/simulated_fmri_snr{snr}.nii.gz')
