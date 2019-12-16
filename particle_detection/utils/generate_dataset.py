import numpy as np
import os.path as osp
import sys

sys.path.insert(0, '../../')
from particle_detection.src.config import cfg
from sklearn.preprocessing import StandardScaler
import os
import yaml
from sklearn.externals import joblib
from extract_scan import extract_scan
from joblib import Parallel, delayed
import multiprocessing
import tqdm
import shutil
import argparse


def generate_dataset(sensor, set, particle, features, parameters, voxel_size=0.2, custom_name='', suffix=''):
    parameters['datasets'] = []
    dataset_name = set
    if set is 'training':

        parameters['shuffle'] = True

        if 'stereo' in sensor or parameters['lidar_cropped']:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 0,
                'map_to_vel_z': 2
            }
            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['11-smoke', -1, -1],
                    ['12-smoke', -1, -1],
                    ['13-smoke', -1, -1],
                    ['17-smoke', -1, -1],
                    ['9-smoke', -1, -1],
                    # ['smoke_bush', -1, -1],
                    ['smoke_car_back', -1, -1]
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    # ['3-dust', 467, 510],
                    ['2-dust', 100, 150],
                    ['8-dust', 100, 150],
                    ['4-dust', 100, 150],
                    ['7-dust', 100, 150]
                ]
        else:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 10,
                'map_to_vel_z': 2
            }
            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['11-smoke', -1, -1],
                    ['12-smoke', -1, -1],
                    ['13-smoke', -1, -1],
                    ['17-smoke', -1, -1],
                    ['9-smoke', -1, -1],
                    ['smoke_car_back', -1, -1]
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    ['2-dust', -1, -1],
                    ['8-dust', -1, -1],
                    ['4-dust', -1, -1],
                    ['7-dust', -1, -1]
                ]

    if set is 'validation':
        parameters['shuffle'] = False
        if 'stereo' in sensor or parameters['lidar_cropped']:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 0,
                'map_to_vel_z': 2
            }
            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    # ['smoke_car', -1, 300],
                    ['19-smoke', -1, 300],
                    # ['12-smoke', -1, -1],
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    # ['3-dust', 511, 527],
                    ['3-dust', -1, 300]
                ]
        else:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 10,
                'map_to_vel_z': 2
            }
            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['19-smoke', -1, 300],
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    ['3-dust', -1, 300]
                ]

    if set is 'testing':
        parameters['shuffle'] = False
        if 'stereo' in sensor or parameters['lidar_cropped']:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 0,
                'map_to_vel_z': 2
            }

            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['10-smoke', -1, -1],
                    ['15-smoke', -1, -1],
                    ['16-smoke', -1, -1],
                    ['18-smoke', -1, -1],
                    ['smoke_bush', -1, -1],
                    ['smoke_car', -1, -1],
                    ['smoke_car_back_far', -1, -1]
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    # ['2-dust', 436, 465],
                    ['1-dust', -1, -1],
                    ['5-dust', -1, -1],
                    ['6-dust', -1, -1],
                ]
        else:
            parameters['map_config'] = {
                'map_size_x': 80,
                'map_size_y': 80,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 40,
                'map_to_vel_y': 40,
                'map_to_vel_z': 2
            }

            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['10-smoke', -1, -1],
                    ['15-smoke', -1, -1],
                    ['16-smoke', -1, -1],
                    ['18-smoke', -1, -1],
                    # ['smoke_car_back_far', -1, -1]
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    ['1-dust', -1, -1],
                    ['5-dust', -1, -1],
                    ['6-dust', -1, -1],
                    # ['smoke_bush', -1, -1]
                ]

    if set is 'visualisation':
        parameters['shuffle'] = False
        if 'stereo' in sensor or parameters['lidar_cropped']:
            parameters['map_config'] = {
                'map_size_x': 20,
                'map_size_y': 20,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 10,
                'map_to_vel_y': 0,
                'map_to_vel_z': 2
            }

            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['10-smoke', 200, 250],
                    ['15-smoke', 200, 250],
                    ['16-smoke', 200, 250],
                    ['18-smoke', 200, 250],
                    # ['smoke_bush', 200, 250],
                    # ['smoke_car', 200, 250],
                    ['smoke_car_back_far', 200, 250],

                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    # ['2-dust', 436, 465],
                    ['1-dust', 200, 250],
                    ['5-dust', 200, 250],
                    ['6-dust', 200, 250],
                ]
        else:
            parameters['map_config'] = {
                'map_size_x': 40,
                'map_size_y': 40,
                'map_size_z': 3,
                'voxel_size_x': voxel_size,
                'voxel_size_y': voxel_size,
                'voxel_size_z': voxel_size,
                'voxel_pt_count': 35,
                'map_to_vel_x': 20,
                'map_to_vel_y': 20,
                'map_to_vel_z': 2
            }

            if particle is 'smoke' or particle is 'both':
                dataset_name += '_smoke'
                parameters['datasets'] += [
                    ['10-smoke', 200, 250],
                    ['15-smoke', 200, 250],
                    ['16-smoke', 200, 250],
                    ['18-smoke', 200, 250],
                    ['smoke_bush', 200, 250],
                    ['smoke_car', 200, 250],
                    ['smoke_car_back_far', 200, 250]
                    # ['smoke_bush', 200, 250],
                ]
            if particle is 'dust' or particle is 'both':
                dataset_name += '_dust'
                parameters['datasets'] += [
                    ['1-dust', 200, 250],
                    ['5-dust', 200, 250],
                    ['6-dust', 200, 250],
                ]

    parameters['features'] = features
    if custom_name == '':
        for f_dataset_name in features:
            dataset_name += '_' + f_dataset_name
        dataset_name += suffix
    else:
        dataset_name = custom_name

    print('\n\nGenerating %s' % dataset_name)
    if sensor == 'lidar_hc':
        dataset_dir = osp.join(cfg.DATASETS_DIR, 'hc_datasets', dataset_name)
    elif sensor == 'lidar':
        dataset_dir = osp.join(cfg.DATASETS_DIR, 'dl_datasets', dataset_name)
    else:
        dataset_dir = osp.join(cfg.DATASETS_DIR, 'st_datasets', dataset_name)

    # If dataset exists already erase it
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    os.makedirs(osp.join(dataset_dir, 'scan_voxels'))
    os.makedirs(osp.join(dataset_dir, 'scan_pcls'))
    if parameters['shuffle']:
        os.makedirs(osp.join(dataset_dir, 'scan_voxels_tmp'))
        os.makedirs(osp.join(dataset_dir, 'scan_pcls_tmp'))

    # datasets = [osp.join(datadir,d) for d in datasets]

    parameters['dataset_size'] = 0
    scaler = StandardScaler()

    scan_id = 0

    for idx, [d, start_id, end_id] in enumerate(parameters['datasets']):

        print("Generating voxels for dataset: %s..." % d)
        if sensor == 'stereo':
            if parameters['stereo_sampled']:
                pcls = np.load(
                    osp.join(cfg.RAW_DATA_DIR, d, 'multi_pcl_sync_sampled_labelled_projected_refined.npy'))
            else:
                pcls = np.load(osp.join(cfg.RAW_DATA_DIR, d, 'multi_pcl_sync_labelled_projected_refined.npy'))
        else:
            if parameters['lidar_cropped']:
                pcls = np.load(
                    osp.join(cfg.RAW_DATA_DIR, d, 'pcl_labeled_spaces_converted_sync_cropped_refined.npy'))
            else:
                pcls = np.load(osp.join(cfg.RAW_DATA_DIR, d, 'pcl_labeled_spaces_converted_sync.npy'))

        if start_id < 0:
            start_id = 0

        if end_id < 0 or (end_id - start_id) > parameters['max_scans']:
            if start_id + parameters['max_scans'] < pcls.shape[0]:
                end_id = start_id + parameters['max_scans']
            else:
                end_id = pcls.shape[0]

        pcls = pcls[start_id:end_id]
        if sensor == 'stereo':
            for pcl in pcls:
                # Shuffle point order because they are organised as image pixel in the pcl
                np.random.shuffle(pcl)

        voxels = []
        coords = []
        labels = []
        raw_pcl = []

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(extract_scan)(pcl, parameters['features'], parameters['map_config'], sensor) for pcl in
            tqdm.tqdm(
                pcls, total=pcls.shape[0], desc='Generating Voxel Data:', ncols=80, leave=False))
        for buffer in results:
            voxels.append(buffer[0])
            coords.append(buffer[1])
            labels.append(buffer[2])
            raw_pcl.append(buffer[3])
        del results
        del pcls

        print("Post Processing dataset: %s" % d)

        # if parameters['shuffle']:
        #     voxels = np.asarray(voxels)
        #     labels = np.asarray(labels)
        #     coords = np.asarray(coords)
        #     raw_pcl = np.asarray(raw_pcl)
        #     p = np.random.permutation(voxels.shape[0])
        #     voxels = voxels[p]
        #     coords = coords[p]
        #     labels = labels[p]
        #     raw_pcl = raw_pcl[p]

        if sensor == 'lidar_hc':
            parameters['features_size'] = voxels[0].shape[1]
        else:
            parameters['features_size'] = voxels[0].shape[2]

        # Process each scan
        for v, c, l, pcl in zip(voxels, coords, labels, raw_pcl):

            if not parameters['separate_fog_dust']:
                # Check if fog or dust file
                particle_label = np.max(l)
                # Check that there is at least one particle label in scan
                if particle_label > 0:

                    if (parameters['fog_only'] and particle_label == 1) or (
                            parameters['dust_only'] and particle_label == 2):
                        v = v[l == 0]
                        c = c[l == 0]
                        l = l[l == 0]
                        if sensor == 'lidar_hc' or sensor == 'lidar':
                            pcl = pcl[pcl[:, 5] == 0, :]
                        else:
                            pcl = pcl[pcl[:, 4] == 0, :]
                    else:
                        # If so, overwrites the particle label to 1
                        output_label = 1
                        l[l == particle_label] = output_label
                        if sensor == 'lidar_hc' or sensor == 'lidar':
                            pcl[pcl[:, 5] == particle_label, 5] = output_label
                        else:
                            pcl[pcl[:, 4] == particle_label, 4] = output_label

            # Update scaler for dataset
            if sensor == 'lidar_hc':
                scaler.partial_fit(v)
            else:
                scaler.partial_fit(v.reshape(-1, v.shape[2]))
            if parameters['shuffle']:
                tmp = '_tmp'
            else:
                tmp = ''
            np.save(osp.join(dataset_dir, 'scan_pcls' + tmp, str(scan_id)), pcl)
            np.save(osp.join(dataset_dir, 'scan_voxels' + tmp, 'voxels_' + str(scan_id)), v)
            np.save(osp.join(dataset_dir, 'scan_voxels' + tmp, 'labels_' + str(scan_id)), l)
            np.save(osp.join(dataset_dir, 'scan_voxels' + tmp, 'coords_' + str(scan_id)), c)

            parameters["dataset_size"] += v.shape[0]
            scan_id += 1

    parameters["nb_scans"] = scan_id

    if parameters['shuffle']:
        perm = np.random.permutation(parameters["nb_scans"])
        for i, p in tqdm.tqdm(
                enumerate(perm), total=parameters["nb_scans"], desc='Shuffling scans Data:', ncols=80,
                leave=False):
            shutil.copyfile(osp.join(dataset_dir, 'scan_pcls_tmp', str(i) + '.npy'),
                            osp.join(dataset_dir, 'scan_pcls', str(p) + '.npy'))
            shutil.copyfile(osp.join(dataset_dir, 'scan_voxels_tmp', 'voxels_' + str(i) + '.npy'),
                            osp.join(dataset_dir, 'scan_voxels', 'voxels_' + str(p) + '.npy'))
            shutil.copyfile(osp.join(dataset_dir, 'scan_voxels_tmp', 'labels_' + str(i) + '.npy'),
                            osp.join(dataset_dir, 'scan_voxels', 'labels_' + str(p) + '.npy'))
            shutil.copyfile(osp.join(dataset_dir, 'scan_voxels_tmp', 'coords_' + str(i) + '.npy'),
                            osp.join(dataset_dir, 'scan_voxels', 'coords_' + str(p) + '.npy'))

        shutil.rmtree(osp.join(dataset_dir, 'scan_pcls_tmp'))
        shutil.rmtree(osp.join(dataset_dir, 'scan_voxels_tmp'))

    # Save parameters in yaml file
    with open(osp.join(dataset_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(parameters, f, default_flow_style=False)

    # Save scaler
    joblib.dump(scaler, osp.join(dataset_dir, 'scaler.pkl'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--set_id', type=int)
    parser.add_argument('--particle_id', type=int)
    parser.add_argument('--feature_id', type=int)
    parser.add_argument('--sensor', type=str,)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--custom_name', type=str, default='')
    parser.add_argument('--voxel_size', type=float, default=0.2)
    parser.add_argument('--stereo_sampled', action='store_false')
    parser.add_argument('--lidar_cropped', action='store_true')
    parser.add_argument('--fog_only', action='store_true')
    parser.add_argument('--dust_only', action='store_true')
    parser.add_argument('--separate_fog_dust', action='store_true')
    parser.add_argument('--manual_seed', type=int)

    args = parser.parse_args()

    sets = [
        'training',
        'validation',
        'testing',
        'visualisation'
    ]

    particles = [
        'smoke',
        'dust',
        'both'
    ]

    if args.sensor == 'stereo':
        features = [
            ['relpos', 'rgb'],
            ['relpos'],
            ['rgb'],
        ]
    elif args.sensor == 'lidar':
        features = [
            ['int', 'relpos', 'echo'],
            ['int'],
            ['echo'],
            ['relpos'],
            ['int', 'relpos'],
            ['int', 'echo'],
            ['relpos', 'echo'],
        ]
    elif args.sensor == 'lidar_hc':
        features = [
            ['intmean', 'intvar'],
            ['roughness', 'slope'],
            ['echo'],
            ['intmean', 'intvar', 'roughness', 'slope'],
            ['intmean', 'intvar', 'echo'],
            ['roughness', 'slope', 'echo'],
            ['intmean', 'intvar', 'roughness', 'slope', 'echo'],
        ]
    else:
        sys.exit('Unrecognised sensor type')

    parameters = {
        "separate_fog_dust": args.separate_fog_dust,  # consider fog and dust as two different labels (fog=1, dust=2)
        "max_scans": 1000,
        "sensor": args.sensor,
        "stereo_sampled": args.stereo_sampled,
        "lidar_cropped": args.lidar_cropped,
        "fog_only": args.fog_only,
        "dust_only": args.dust_only,
    }
    if args.manual_seed is not None:
        np.random.seed(args.manual_seed)
        parameters['manual_seed'] = args.manual_seed
    else:
        parameters['manual_seed'] = 'None'

    # # Force full pcl when visulisation set
    # if sets[args.set_id] == 'visualisation':
    #     parameters['stereo_sampled'] = False

    generate_dataset(args.sensor, sets[args.set_id], particles[args.particle_id], features[args.feature_id], parameters, voxel_size=args.voxel_size, custom_name=args.custom_name,suffix=args.suffix)

    # voxel_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
    # names = ['_10cm', '_15cm', '_20cm', '_25cm', '_30cm']
    #
    # for i, n in zip(voxel_sizes,names):
    #     generate_dataset(args.sensor, sets[args.set_id], particles[args.particle_id], features[args.feature_id],
    #                      parameters, voxel_size=i, custom_name='12-smoke'+n)

    # generate_dataset('lidar', sets[3], particles[2], features[0], parameters, voxel_size=0.2, custom_name='6_dust_start')