import os, sys, time
import argparse
import numpy as np
import pickle
from typing import Dict, Any
import json
import copy



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src-annot',                      type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1')
    parser.add_argument('--tgt-annot',                      type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1')

    '''
    parser.add_argument('--src-annot-hd',                   type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1/90008d1_03_input_face_hd.json')
    parser.add_argument('--src-face-lr',                    type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1/90008d1_01_input_face.png')
    parser.add_argument('--src-face-mesh',                  type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1/90008d1_0_hrn_mid_mesh.obj')
    parser.add_argument('--src-vertex-ids-pixels-map',      type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1/90008d1_vertex_ids_pixels_map.npy')
    parser.add_argument('--src-pixel-coords-by-vertex-ids', type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90008/90008d1/90008d1_pixel_coords_by_vertex_ids.pkl')

    parser.add_argument('--tgt-face-hd',                    type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1/90009d1_03_input_face_hd.png')
    parser.add_argument('--tgt-face-lr',                    type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1/90009d1_01_input_face.png')
    parser.add_argument('--tgt-face-mesh',                  type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1/90009d1_0_hrn_mid_mesh.obj')
    parser.add_argument('--tgt-vertex-ids-pixels-map',      type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1/90009d1_vertex_ids_pixels_map.npy')
    parser.add_argument('--tgt-pixel-coords-by-vertex-ids', type=str, default='/datasets1/bjgbiesseck/ND-Twins-2009-2010/images_format=PNG_HRN_3D_Reconstruction/90009/90009d1/90009d1_pixel_coords_by_vertex_ids.pkl')
    '''

    args = parser.parse_args()
    return args


def load_json(file_path: str) -> Dict[Any, Any]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error: Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def save_json(data: Dict[Any, Any], file_path: str) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=1)
    except TypeError as e:
        raise TypeError(f"Error: Data is not serializable to JSON: {e}")
    except OSError as e:
        raise OSError(f"Error writing to file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def load_pickle(file_path: str) -> Any:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error: Failed to unpickle file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")



def main_transfer_annotation(args):
    args.src_annot = args.src_annot.rstrip('/')
    args.tgt_annot = args.tgt_annot.rstrip('/')
    
    src_dir_name = args.src_annot.split('/')[-1]
    tgt_dir_name = args.tgt_annot.split('/')[-1]

    # LOAD SOURCE DATA
    print('Loading source data...')
    path_src_annot_hd                   = os.path.join(args.src_annot, src_dir_name+'_03_input_face_hd.json')
    src_annot_hd                        = load_json(path_src_annot_hd)
    path_src_vertex_ids_pixels_map      = os.path.join(args.src_annot, src_dir_name+'_vertex_ids_pixels_map.npy')
    src_vertex_ids_pixels_map           = np.load(path_src_vertex_ids_pixels_map)[0]
    path_src_pixel_coords_by_vertex_ids = os.path.join(args.src_annot, src_dir_name+'_pixel_coords_by_vertex_ids.pkl')
    src_pixel_coords_by_vertex_ids = load_pickle(path_src_pixel_coords_by_vertex_ids)


    # LOAD TARGET DATA
    print('Loading target data...')
    path_tgt_vertex_ids_pixels_map      = os.path.join(args.tgt_annot, tgt_dir_name+'_vertex_ids_pixels_map.npy')
    tgt_vertex_ids_pixels_map           = np.load(path_tgt_vertex_ids_pixels_map)[0]
    path_tgt_pixel_coords_by_vertex_ids = os.path.join(args.tgt_annot, tgt_dir_name+'_pixel_coords_by_vertex_ids.pkl')
    tgt_pixel_coords_by_vertex_ids      = load_pickle(path_tgt_pixel_coords_by_vertex_ids)


    src_imageHeight_hd, src_imageWidth_hd = src_annot_hd['imageHeight'], src_annot_hd['imageWidth']
    tgt_imageHeight_lr, tgt_imageWidth_lr = [224, 224]
    anoot_scaleHeight = tgt_imageHeight_lr/src_imageHeight_hd
    anoot_scaleWidth  = tgt_imageWidth_lr/src_imageWidth_hd
    print('anoot_scaleHeight:', anoot_scaleHeight, '    anoot_scaleWidth:', anoot_scaleWidth)


    print('Transfering annotation...')
    tgt_annot_hd = copy.deepcopy(src_annot_hd)
    for idx_shape, (src_shape, tgt_shape) in enumerate(zip(src_annot_hd['shapes'], tgt_annot_hd['shapes'])):
        src_shape_points = src_shape['points']
        tgt_points = []
        for idx_point, src_shape_point in enumerate(src_shape_points):
            src_shape_point = np.array(src_shape_point)
            src_shape_point = src_shape_point*anoot_scaleHeight
            print('src_shape_point:', src_shape_point)
            src_vertex_idx = src_vertex_ids_pixels_map[int(round(src_shape_point[1])), int(round(src_shape_point[0]))]
            print('src_vertex_idx:', src_vertex_idx)
            new_tgt_shape_point = tgt_pixel_coords_by_vertex_ids[src_vertex_idx]
            if not new_tgt_shape_point is None:
                print('new_tgt_shape_point:', new_tgt_shape_point, '    type(new_tgt_shape_point):', type(new_tgt_shape_point))
                tgt_shape_point = [new_tgt_shape_point[1]/anoot_scaleHeight, new_tgt_shape_point[0]/anoot_scaleHeight]
                tgt_points.append(tgt_shape_point)
            # sys.exit(0)

        tgt_shape['points'] = tgt_points
    tgt_annot_hd['imagePath'] = tgt_dir_name + '_03_input_face_hd.png'

    path_tgt_annot_hd = os.path.join(args.tgt_annot, tgt_dir_name+'_03_input_face_hd.json')
    print(f'Saving transfered annotation: \'{path_tgt_annot_hd}\'')
    save_json(tgt_annot_hd, path_tgt_annot_hd)

    print('\n----------')
    print('Finish!')


if __name__ == '__main__':
    args = parse_args()
    main_transfer_annotation(args)