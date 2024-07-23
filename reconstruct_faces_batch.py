import os, sys, time
from models.hrn import Reconstructor
import cv2
from tqdm import tqdm
import argparse
import numpy as np

# Bernardo
def find_image_files(folder_path):
    img_ext = ['.jpg', '.jpeg', '.png']
    found_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            file_lower = file.lower()
            for ext in img_ext:
                if file_lower.endswith(ext):
                    found_files.append(os.path.join(root, file))
                    break
    found_files.sort()
    return found_files


# Bernardo
def search_save_inappropriate_files(files_paths, str_search='aaa', path_save_list='inappropriate_files.txt'):
    num_inappropriate_found = 0
    inappropriate_files = []
    inappropriate_found = False
    if len(files_paths) > 0:
        print('\nSearching inappropriate_files...')
        for i, file_path in enumerate(files_paths):
            if str_search in file_path:
                inappropriate_found = True
                num_inappropriate_found += 1
                print('Inappropriate file found:', file_path)
                inappropriate_files.append(file_path)
        
        if inappropriate_found:
            print('\nSaving inappropriate files paths')
            with open(path_save_list, 'w') as file1:
                for i, inapp_file in enumerate(inappropriate_files):
                    file1.write(inapp_file + '\n')
                    file1.flush()
    return inappropriate_found, num_inappropriate_found


# Bernardo
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


# Bernardo
def get_batches_indices(list_values, batch_size):
    begin_batch = []
    end_batch = []
    num_batches = int(np.ceil(len(list_values) / batch_size))

    for i_batch in range(0, num_batches):
        begin_batch.append(i_batch*batch_size)
        end_batch.append(min((i_batch*batch_size)+batch_size, len(list_values)))

    return begin_batch, end_batch



def run_hrn(args):
    params = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--epoch', args.epoch,
    ]

    reconstructor = Reconstructor(params)

    # names = sorted([name for name in os.listdir(args.input_root) if '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])
    print(f'\nSearching image files in \'{args.input_root}\'...')
    names = find_image_files(args.input_root) 
    # for i, name in enumerate(names):
    #     print(i, '- name:', name)
    # print('len(names):', len(names))
    # sys.exit(0)

    # Bernardo
    if len(names) == 0:
        raise Exception(f'No images found in \'{args.input_root}\'')

    # Bernardo
    str_search = 'render.jpg'
    path_inappropriate_list = 'inappropriate_files=' + str_search + '.txt'
    inappropriate_found, num_inappropriate_found = search_save_inappropriate_files(names, str_search, path_inappropriate_list)
    if inappropriate_found:
        raise Exception(str(num_inappropriate_found)+' inappropriate files found. List of files saved in \''+str(path_inappropriate_list)+'\'')

    if not os.path.isdir(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    # print('predict', args.input_root)


    begin_parts, end_parts = get_parts_indices(names, args.div)
    names_part = names[begin_parts[args.part]:end_parts[args.part]]
    print('begin_parts:', begin_parts, '    end_parts:', end_parts)
    # sys.exit(0)


    begin_index_str = 0
    end_index_str = len(names_part)
    if args.str_begin != '':
        print('Searching str_begin \'' + args.str_begin + '\' ...  ')
        for x, name in enumerate(names_part):
            if args.str_begin in name:
                begin_index_str = x
                print('found at', begin_index_str)
                break

    if args.str_end != '':
        print('Searching str_end \'' + args.str_end + '\' ...  ')
        for x, name in enumerate(names_part):
            if args.str_end in name:
                end_index_str = x+1
                print('found at', end_index_str)
    
    print('\n------------------------')
    print(f'part {args.part}/{args.div}')
    print('begin_index_str:', begin_index_str)
    print('end_index_str:', end_index_str)
    print('------------------------\n')
    names_part = names_part[begin_index_str:end_index_str]
    # sys.exit(0)


    start_idx = 0
    if args.find_substring != '':
        for i, n in enumerate(names_part):
            if args.find_substring in n:
                start_idx = i
                break
    # print('start_idx:', start_idx)
    # sys.exit(0)

    batches_begin_idx, batches_end_idx = get_batches_indices(names_part, args.batch)

    # for ind in range(start_idx, len(names_part), ):
    for idx_batch, (begin_batch, end_batch) in enumerate(zip(batches_begin_idx, batches_end_idx)):
        print(f'idx_batch: {idx_batch}/{len(batches_begin_idx)}')
        names_batch = names_part[begin_batch:end_batch]
        print('names_batch:', names_batch)

        t1 = time.time()
        output_batch = reconstructor.predict_batch(args, names_batch, visualize=True)
        # print('output_batch:', output_batch)
        # print('output_batch.shape:', output_batch.shape)
        
        print(f'Elapsed time: {time.time() - t1} sec')
        print('----------------')

        '''
        # print(f'divs: {args.div}    part: {args.part}    ind: {ind}/{len(names_part)-1}')
        print('name:', name)
        # save_name = os.path.splitext(name)[0]                                         # original
        save_name = os.path.splitext(name)[0].replace(args.input_root, '').strip('/')   # Bernardo
        sub_dirs, save_name = '/'.join(save_name.split('/')[:-1]), save_name.split('/')[-1]
        print('args.output_root:', args.output_root)
        print('sub_dirs:', sub_dirs)
        print('save_name:', save_name)
        out_dir = os.path.join(args.output_root, sub_dirs, save_name)
        print('out_dir:', out_dir)
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(name)
        # print('os.path.join(args.input_root, name):', os.path.join(args.input_root, name))

        t1 = time.time()
        output = reconstructor.predict(args, img, visualize=True, out_dir=out_dir, save_name=save_name)
        print(f'save results: {time.time() - t1} sec')
        print('----------------')
        # sys.exit(0)
        '''

    print('results are saved to:', args.output_root)


def run_mvhrn(args):
    params = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--epoch', args.epoch,
    ]

    reconstructor = Reconstructor(params)

    names = sorted([name for name in os.listdir(args.input_root) if
                    '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])
    os.makedirs(args.output_root, exist_ok=True)

    print('predict', args.input_root)

    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)
    img_list = []
    for ind, name in enumerate(names):
        img = cv2.imread(os.path.join(args.input_root, name))
        img_list.append(img)
        # output = reconstructor.predict_base(img, save_name=save_name, out_dir=out_dir)
    output = reconstructor.predict_multi_view(img_list, visualize=True, out_dir=out_dir)

    print('results are saved to:', args.output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoints_dir', type=str, default='assets/pretrained_models', help='models are saved here')
    parser.add_argument('--name', type=str, default='hrn_v1.1',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--epoch', type=str, default='10', help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--input_type', type=str, default='single_view',  # or 'multi_view'
                        help='reconstruct from single-view or multi-view')
    parser.add_argument('--input_root', type=str, default='./assets/examples/single_view_image',
                        help='directory of input images')
    parser.add_argument('--input_lmks', type=str, default='./assets/examples/single_view_image_lmks',
                        help='')
    parser.add_argument('--output_root', type=str, default='./assets/examples/single_view_image_results_BATCH',
                        help='directory for saving results')

    # Bernardo
    parser.add_argument('--batch', default=4, type=int, help='')

    parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')

    # Bernardo
    parser.add_argument('--div', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    # Bernardo
    parser.add_argument('--find_substring', type=str, default='', help='directory for saving results')
    parser.add_argument('--no_face_align', action='store_true')
    parser.add_argument('--save_only_sampled', action='store_true')
    parser.add_argument('--no_reconstruction', action='store_true')

    args = parser.parse_args()


    assert args.part < args.div, f'Error, args.part ({args.part}) >= args.div ({args.div}). args.part must be less than args.div!'


    if args.input_type == 'multi_view':
        run_mvhrn(args)
    else:
        run_hrn(args)

    print('\nFinished!\n')
