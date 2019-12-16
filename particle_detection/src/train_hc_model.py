#!/usr/bin/env python

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from particle_dataset import ParticleDataset
from config import cfg
import os
from datetime import datetime
import os.path as osp
import yaml
from sklearn.externals import joblib
import numpy as np
import argparse


class Classifier(object):
    def __init__(self, model=None, search_params=None):
        self.model = model
        self.search_params = search_params


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    labels = []

    for i, sample in enumerate(batch):
        voxel_features.append(sample['inputs'])

        voxel_coords.append(
            np.pad(sample['coords'], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        labels.append(sample['labels'])

    return {
        'inputs': np.concatenate(voxel_features),
        'coords': np.concatenate(voxel_coords),
        'labels': np.concatenate(labels)}


def train_model(classifier_name, particles, features, output_model, train_data=None, val_data=None):
    datasets_dir = osp.join(cfg.DATASETS_DIR, 'hc_datasets')
    logs_dir = osp.join(cfg.LOG_DIR, 'hc_logs')

    # output_model = 'smallest_smoke_dust_all'

    # train_data = 'training_smallest_smoke_dust_intmean_intvar_roughness_slope_echo'
    # val_data = 'validation_smoke_dust_intmean_intvar_roughness_slope_echo'
    if not train_data:
        train_data = 'training_'+particles+'_intmean_intvar_roughness_slope_echo'
    if not val_data:
        val_data = 'training_'+particles+'_intmean_intvar_roughness_slope_echo'

    now = datetime.now()

    with open(osp.join(datasets_dir, train_data, 'config.yaml'), 'r') as f:
        dataset_params = yaml.load(f, Loader=yaml.SafeLoader)

    # Parameters
    parameters = {
        "train_data": train_data,
        "val_data": val_data,
        "features": features,
        "map_config": dataset_params['map_config'],
    }

    train_data = osp.join(datasets_dir, train_data)
    val_data = osp.join(datasets_dir, val_data)

    output_path = osp.join(logs_dir, now.strftime('%Y%m%d_%H%M%S') + '_' + output_model)

    # Save training parameters
    os.makedirs(output_path)

    # Datasets
    train_loader = DataLoader(ParticleDataset(dataset_dir=train_data, use_dataset_scaler=True), batch_size=1,
                              shuffle=False, num_workers=4, collate_fn=detection_collate)

    val_loader = DataLoader(ParticleDataset(dataset_dir=val_data, scaler=train_loader.dataset.scaler), batch_size=1,
                            shuffle=False, num_workers=4, collate_fn=detection_collate)

    inputs = []
    y_target = []

    for sample in train_loader:
        selected_inputs = np.array([]).reshape(sample['inputs'].shape[0], 0)

        if 'roughness' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 0].reshape(-1, 1)), axis=1)
        if 'slope' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 1].reshape(-1, 1)), axis=1)
        if 'intmean' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 2].reshape(-1, 1)), axis=1)
        if 'intmean' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 3].reshape(-1, 1)), axis=1)
        if 'echo' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 4:7]), axis=1)
        inputs.append(selected_inputs)
        y_target.append(sample['labels'])

    inputs = np.concatenate(inputs)
    y_target = np.concatenate(y_target)

    # Models
    # clf = Classifier()

    # if feature_selection:
    #     trainset.select_features('TreeBased')
    # features = trainset.features
    # mu, sigma = trainset.mu, trainset.sigma

    # X_train = trainset.data
    # y_train = trainset.labels

    if classifier_name == 'random_forest':
        clf = Classifier(RandomForestClassifier(max_features=inputs.shape[1], n_estimators=100, max_depth=10))
    elif classifier_name == 'SVM_linear':
        clf = Classifier(SVC(kernel='linear'))  # SVM Linear kernel
    elif classifier_name == 'SVM_RBF':
        clf = Classifier(SVC(kernel='rbf'))  # SVM RBF kernel
    else:
        clf = Classifier(GaussianNB())  # Naive Bayes

    # Uncomment this to perform random search
    # classifiers['random_forest'].search_params = {'n_estimators': [3, 10, 30, 100, 200],
    #                                               'max_features': range(1, len(features)+1),
    #                                               'max_depth': [1, 5, 10, 20]}
    #
    # classifiers['SVM'].search_params = {'kernel': ['rbf', 'linear'], 'C': [0.025, 0.1, 0.5, 1],
    #                                     'gamma': ['auto', 2]}

    # Save model parameters
    with open(osp.join(output_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(parameters, f, default_flow_style=False)
    joblib.dump(train_loader.dataset.scaler, osp.join(output_path, 'scaler.pkl'))

    # Start training

    print('Training model...')

    # Uncomment this to perform random search
    # classifiers['random_forest'].search_params = {'n_estimators': [3, 10, 30, 100, 200],
    #                                               'max_features': range(1, len(features)+1),
    #                                               'max_depth': [1, 5, 10, 20]}
    #
    # classifiers['SVM'].search_params = {'kernel': ['rbf', 'linear'], 'C': [0.025, 0.1, 0.5, 1],
    #                                     'gamma': ['auto', 2]}

    # Model fine-tuning
    if clf.search_params:
        param_search = RandomizedSearchCV(clf.model, clf.search_params, cv=5, scoring='neg_mean_squared_error')
        param_search.fit(inputs, y_target)

        final_model = param_search.best_estimator_  # Get full best estimator (if refit=true it retains it on the whole training set)

        print('Grid search evaluation')
        cvres = param_search.cv_results_  # Get evaluation score

        for mean_score, params in sorted(zip(cvres["mean_test_score"], cvres["params"]), reverse=True):
            print(np.sqrt(-mean_score), params)
        clf.model = param_search.best_estimator_  # Get full best estimator (if refit=true it retains it on the whole training set)
    else:
        clf.model.fit(inputs, y_target)  # Train classifier

    clf.model.fit(inputs, y_target)

    joblib.dump(clf.model, osp.join(output_path, 'model.pkl'))

    # Validation

    print('Validating model...')

    inputs = []
    y_target = []

    # Concatenate scans and pick features
    for sample in val_loader:
        selected_inputs = np.array([]).reshape(sample['inputs'].shape[0], 0)

        if 'roughness' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 0].reshape(-1, 1)), axis=1)
        if 'slope' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 1].reshape(-1, 1)), axis=1)
        if 'intmean' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 2].reshape(-1, 1)), axis=1)
        if 'intvar' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 3].reshape(-1, 1)), axis=1)
        if 'echo' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 4:7]), axis=1)
        inputs.append(selected_inputs)
        y_target.append(sample['labels'])

    inputs = np.concatenate(inputs)
    y_target = np.concatenate(y_target)

    y_pred = clf.model.predict(inputs)

    with open(osp.join(output_path, 'val_results.txt'), 'w') as f:

        f.write('Validation Results:\n')
        # Compute performance scores

        print('Confusion Matrix')
        f.write("Confusion Matrix\n")

        cnf_matrix = confusion_matrix(y_target, y_pred).astype(np.float64)  # Compute confusion matrix
        cnf_matrix /= cnf_matrix.sum(1, keepdims=True)  # put into ratio
        cnf_matrix = cnf_matrix[:2, :2]
        np.set_printoptions(precision=2)
        print(cnf_matrix)
        f.write(str(cnf_matrix))
        f.write('\n')

        # Can only use this if both classes are at least predicted once
        if len(np.unique(y_pred)) > 1:
            print('Classification Report')
            f.write('Classification Report\n')

            cr = classification_report(y_target, y_pred, digits=2)
            print(cr)
            f.write(cr)
            f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--id', type=int, default=0, help='features id')
    parser.add_argument('--prefix', type=str, default='0', help='prefix')

    args = parser.parse_args()

    particles = [
        # 'smoke',
        # 'dust',
        'smoke_dust'
    ]

    features = [
        ['intmean', 'intvar', 'roughness', 'slope', 'echo'],
        ['intmean', 'intvar'],
        ['roughness', 'slope'],
        ['echo'],
        ['intmean', 'intvar', 'roughness', 'slope'],
        ['intmean', 'intvar', 'echo'],
        ['roughness', 'slope', 'echo'],
    ]

    classifier = [
        'random_forest',
        'SVM_RBF',
        'SVM_linear',
        # 'naive_bayes'
    ]



    # for c in classifier:
    #     for p in particles:
    #         for f in features:

    p = particles[0]
    f = features[4]
    c = classifier[args.id]

    if c == 'SVM_RBF' or c == 'SVM_linear' or c == 'random_forest':
        train_set = 'training_tiny_smoke_dust_intmean_intvar_roughness_slope_echo'
        name_desc = 'tiny_'
    else:
        train_set = 'training_smoke_dust_intmean_intvar_roughness_slope_echo'
        name_desc = ''

    name = name_desc
    name += p
    for feature_name in f:
        name += '_' + feature_name

    name = c + '_' + name + args.prefix
    train_model(c, p, f, name,train_data=train_set)
