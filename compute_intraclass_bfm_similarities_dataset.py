import os
import sys

import argparse
import random
import socket
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
# from pytorch3d.io import load_obj
# from pytorch3d.loss import chamfer_distance
# from mpl_toolkits.mplot3d import Axes3D

import glob
# from mpl_toolkits.mplot3d import Axes3D
# from pytorch3d.io import load_obj, load_ply
# from pytorch3d.loss import chamfer_distance

from util.load_mats import LoadBFM09, LoadExpBasis


def get_parts_indices(sub_folders, divisions):
    begin_div = []
    end_div = []
    div_size = int(len(sub_folders) / divisions)
    remainder = int(len(sub_folders) % divisions)

    for i in range(0, divisions):
        begin_div.append(i*div_size)
        end_div.append(i*div_size + div_size)
    
    end_div[-1] += remainder

    # print('begin_div:', begin_div)
    # print('end_div:', end_div)
    # sys.exit(0)
    return begin_div, end_div


def load_sample(file_path):
    if file_path.endswith('_coeffs.npy'):
        bfm_coeffs = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: \'file_path\'")
    return bfm_coeffs
    

def compute_chamfer_distance(points1, points2):
    chamfer_dist = chamfer_distance(points1.unsqueeze(0), points2.unsqueeze(0))
    return chamfer_dist[0]


def compute_cosine_sim(array1, array2, normalize=True):
    if array1.shape[0] == 1:
        array1 = array1[0]
    if array2.shape[0] == 1:
        array2 = array2[0]

    if isinstance(array1, np.ndarray):
         array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
         array2 = torch.from_numpy(array2)

    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    return cos_sim


def compute_cosine_distance(array1, array2, normalize=True):
    if array1.shape[0] == 1:
        array1 = array1[0]
    if array2.shape[0] == 1:
        array2 = array2[0]

    if isinstance(array1, np.ndarray):
         array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
         array2 = torch.from_numpy(array2)
    
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    cos_dist = 1.0 - cos_sim
    return cos_dist


def compute_euclidean_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    eucl_dist = torch.norm(array1 - array2)
    return eucl_dist


def find_files_by_extension(folder_path, extension, ignore_file_with=''):
    matching_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if ignore_file_with == '' or not ignore_file_with in file_path:
                    matching_files.append(file_path)
    return sorted(matching_files)


def rescale_coeffs(bfm_coeffs_split, model_bfm_exp):
    bfm_coeffs_split['id']  *= np.squeeze(model_bfm_exp['idEV'])
    bfm_coeffs_split['exp'] *= np.squeeze(model_bfm_exp['exEV'])
    bfm_coeffs_split['tex'] *= np.squeeze(model_bfm_exp['texEV'])
    return bfm_coeffs_split


def split_coeff(coeffs):
    id_coeffs = coeffs[:, :80]         # face identity    (80)
    exp_coeffs = coeffs[:, 80: 144]    # face expression  (64)
    tex_coeffs = coeffs[:, 144: 224]   # texture          (80)
    angles = coeffs[:, 224: 227]       # face rotation    ( 3)
    gammas = coeffs[:, 227: 254]       # gammas           (27)
    translations = coeffs[:, 254:]     # face translation ( 3)
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


def rescale_coeffs(bfm_coeffs_split, model_bfm_exp):
    bfm_coeffs_split['id']  *= np.squeeze(model_bfm_exp['idEV'])
    bfm_coeffs_split['exp'] *= np.squeeze(model_bfm_exp['exEV'])
    bfm_coeffs_split['tex'] *= np.squeeze(model_bfm_exp['texEV'])
    return bfm_coeffs_split


def main(args):
    assert args.part < args.divs, f'Error, args.part ({args.part}) >= args.divs ({args.divs}), but should be args.part ({args.part}) < args.divs ({args.divs})'

    model_bfm_exp = LoadBFM09(bfm_folder=os.path.dirname(args.bfm_basis), exp_folder=os.path.dirname(args.exp_basis))

    dataset_path = args.input_path.rstrip('/')
    output_path = os.path.join(os.path.dirname(dataset_path), 'cossims_bfm_'+args.metric)
    os.makedirs(output_path, exist_ok=True)

    print('dataset_path:', dataset_path)
    print('Searching subject subfolders...')
    subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    print(f'Found {len(subjects_paths)} subjects!')

    begin_parts, end_parts = get_parts_indices(subjects_paths, args.divs)
    idx_subj_begin, idx_subj_end = begin_parts[args.part], end_parts[args.part]
    num_subjs_part = idx_subj_end - idx_subj_begin 
    print('\nbegin_parts:', begin_parts)
    print('end_parts:  ', end_parts)
    print(f'idx_subj_begin: {idx_subj_begin}    idx_subj_end: {idx_subj_end}')
    print('')
    
    print('Computing distances...\n')
    for idx_subj, subj_path in enumerate(subjects_paths):
        if idx_subj >= idx_subj_begin and idx_subj < idx_subj_end:
            subj_start_time = time.time()

            subj = os.path.basename(subj_path)
            output_subj_path = os.path.join(output_path, subj)
            os.makedirs(output_subj_path, exist_ok=True)

            cossims_id_file_name = 'cossims_bfm_id_'+args.metric+'.npy'
            output_cossims_id_path = os.path.join(output_subj_path, cossims_id_file_name)
            cossims_exp_file_name = 'cossims_bfm_exp_'+args.metric+'.npy'
            output_cossims_exp_path = os.path.join(output_subj_path, cossims_exp_file_name)
            cossims_pose_file_name = 'cossims_bfm_pose_'+args.metric+'.npy'
            output_cossims_pose_path = os.path.join(output_subj_path, cossims_pose_file_name)
            
            print(f'{idx_subj}/{len(subjects_paths)} - Searching subject samples in \'{subj_path}\'')
            ignore_file_with = 'mean_embedding'
            samples_paths = find_files_by_extension(subj_path, args.file_ext, ignore_file_with)

            loaded_samples = [None] * len(samples_paths)
            for idx_sf, sample_path in enumerate(samples_paths):
                print(f'Loading samples - {idx_sf}/{len(samples_paths)}...', end='\r')
                data = load_sample(sample_path)
                loaded_samples[idx_sf] = data
            print('')

            cossim_id_samples_matrix   = -np.ones((len(loaded_samples),len(loaded_samples)), dtype=np.float32)
            cossim_exp_samples_matrix  = -np.ones((len(loaded_samples),len(loaded_samples)), dtype=np.float32)
            cossim_pose_samples_matrix = -np.ones((len(loaded_samples),len(loaded_samples)), dtype=np.float32)
            for i in range(len(loaded_samples)):
                sample1 = loaded_samples[i]
                sample1_split = split_coeff(sample1)
                sample1_split = rescale_coeffs(sample1_split, model_bfm_exp)

                for j in range(i+1, len(loaded_samples)):
                    print(f'    Computing intra-class \'{args.metric}\' distances - i: {i}/{len(loaded_samples)}  j: {j}/{len(loaded_samples)}', end='\r')
                    sample2 = loaded_samples[j]
                    sample2_split = split_coeff(sample2)
                    sample2_split = rescale_coeffs(sample2_split, model_bfm_exp)

                    if args.metric == 'cosine_2d':
                        cossim_id   = compute_cosine_sim(sample1_split['id'],    sample2_split['id'])
                        cossim_exp  = compute_cosine_sim(sample1_split['exp'],   sample2_split['exp'])
                        cossim_pose = compute_cosine_sim(sample1_split['angle'], sample2_split['angle'])

                    cossim_id_samples_matrix[i,j]   = cossim_id
                    cossim_exp_samples_matrix[i,j]  = cossim_exp
                    cossim_pose_samples_matrix[i,j] = cossim_pose
            print('')

            # if not skip_distances_between_samples:
            print(f'    Saving cossims between samples: \'{output_cossims_id_path}\'')
            np.save(output_cossims_id_path, cossim_id_samples_matrix)
            np.save(output_cossims_exp_path, cossim_exp_samples_matrix)
            np.save(output_cossims_pose_path, cossim_pose_samples_matrix)

            subj_elapsed_time = (time.time() - subj_start_time)
            print('    subj_elapsed_time: %.2f sec' % (subj_elapsed_time))
            print('---------------------')
            # sys.exit(0)

    print('\nFinished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bfm_basis', type=str, default='/datasets2/bjgbiesseck/BaselFaceModel/PublicMM1/01_MorphableModel.mat', help='')
    parser.add_argument('--exp_basis', type=str, default='/datasets2/bjgbiesseck/microsoft_Deep3DFaceReconstruction/Exp_Pca.bin', help='')

    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_BFM/1_CASIA-WebFace/imgs_crops_112x112')
    
    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--metric', default='cosine_2d', type=str, help='')
    parser.add_argument('--file_ext', default='_coeffs.npy', type=str, help='')

    parser.add_argument('--dont_replace_existing_files', action='store_true', help='')

    args = parser.parse_args()

    main(args)