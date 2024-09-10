import os, sys
import numpy as np
import argparse

from util.load_mats import LoadBFM09, LoadExpBasis

from scipy.io import loadmat

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bfm_basis', type=str, default='/datasets2/bjgbiesseck/BaselFaceModel/PublicMM1/01_MorphableModel.mat', help='')
    parser.add_argument('--exp_basis', type=str, default='/datasets2/bjgbiesseck/microsoft_Deep3DFaceReconstruction/Exp_Pca.bin', help='')

    parser.add_argument('--sample0', type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_HRN_3D_face_reconstruction/assets/results/KDEF_results/AF02/AF02AFFL/AF02AFFL_coeffs.npy', help='')
    parser.add_argument('--sample1', type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_HRN_3D_face_reconstruction/assets/results/KDEF_results/AF03/AF03AFFL/AF03AFFL_coeffs.npy', help='')

    parser.add_argument('--normalize', action="store_true")


    args = parser.parse_args()
    return args


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


def euclidean_distance(array1, array2, normalize=False):
    array1 = np.array(array1)
    array2 = np.array(array2)
    if normalize:
        array1 = array1 / np.linalg.norm(array1)
        array2 = array2 / np.linalg.norm(array2)
    distance = np.sqrt(np.sum((array1 - array2) ** 2))
    return distance


def cosine_similarity(vec1, vec2, normalize=False):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if normalize:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

    dot_product = np.dot(np.squeeze(vec1), np.squeeze(vec2))
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)



def rescale_coeffs(bfm_coeffs_split, model_bfm_exp):
    bfm_coeffs_split['id']  *= np.squeeze(model_bfm_exp['idEV'])
    bfm_coeffs_split['exp'] *= np.squeeze(model_bfm_exp['exEV'])
    bfm_coeffs_split['tex'] *= np.squeeze(model_bfm_exp['texEV'])
    return bfm_coeffs_split



if __name__ == '__main__':
    args = parse_args()

    # dict_keys(['meanshape', 'meantex', 'idBase', 'idEV', 'exBase', 'exEV', 'texBase', 'texEV', 'tri', 'point_buf', 'tri_mask2', 'keypoints', 'frontmask2_idx', 'skinmask'])
    model_bfm_exp = LoadBFM09(bfm_folder=os.path.dirname(args.bfm_basis), exp_folder=os.path.dirname(args.exp_basis))
    # print('model_bfm_exp.keys()', model_bfm_exp.keys())
    # print("model_bfm_exp['idBase'].shape:", model_bfm_exp['idBase'].shape)
    # print("model_bfm_exp['idEV'].shape:", model_bfm_exp['idEV'].shape)
    # print("model_bfm_exp['exBase'].shape:", model_bfm_exp['exBase'].shape)
    # print("model_bfm_exp['exEV'].shape:", model_bfm_exp['exEV'].shape)
    # print("model_bfm_exp['texBase'].shape:", model_bfm_exp['texBase'].shape)
    # print("model_bfm_exp['texEV'].shape:", model_bfm_exp['texEV'].shape)
    # sys.exit(0)

    bfm0 = np.load(args.sample0)
    bfm1 = np.load(args.sample1)
    # print('bfm0.shape:', bfm0.shape)
    # print('bfm1.shape:', bfm1.shape)

    # 'id', 'exp', 'tex', 'angle', 'gamma', 'trans'
    bfm0_split = split_coeff(bfm0)
    bfm1_split = split_coeff(bfm1)
    # print('bfm0_split:', bfm0_split)

    bfm0_split = rescale_coeffs(bfm0_split, model_bfm_exp)
    bfm1_split = rescale_coeffs(bfm1_split, model_bfm_exp)

    # id_eucl_dist = euclidean_distance(bfm0_split['id'], bfm1_split['id'], normalize=args.normalize)
    # exp_eucl_dist = euclidean_distance(bfm0_split['exp'], bfm1_split['exp'], normalize=args.normalize)
    # pose_eucl_dist = euclidean_distance(bfm0_split['angle'], bfm1_split['angle'], normalize=args.normalize)
    # print('id_eucl_dist:   %.2f' % id_eucl_dist)
    # print('exp_eucl_dist:  %.2f' % exp_eucl_dist)
    # print('pose_eucl_dist: %.2f' % pose_eucl_dist)
    # print('------------')

    id_cossim   = cosine_similarity(bfm0_split['id'], bfm1_split['id'], normalize=args.normalize)
    exp_cossim  = cosine_similarity(bfm0_split['exp'], bfm1_split['exp'], normalize=args.normalize)
    pose_cossim = cosine_similarity(bfm0_split['angle'], bfm1_split['angle'], normalize=args.normalize)
    print('Cosine similarity')
    print('id:   %.2f' % id_cossim)
    print('exp:  %.2f' % exp_cossim)
    print('pose: %.2f' % pose_cossim)
