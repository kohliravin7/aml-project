import os
import argparse
import logging
import time
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

from train_search import darts
from model import NetworkKMNIST as Network
from model import EnsembleModel
from train import train, infer, fit
from darts_bohamiann import BohamiannWorker
from darts_es import ESWorker
from darts_bohb import run_bohb
from datasets import K49, KMNIST
from settings import get, get_darts_args
import utils
import sys

def run_model(config,
         seed=0,
         data_dir='./data',
         genotype_class='PCDARTS',
         num_epochs=20,
         batch_size=get('batch_size'),
         init_channels=get('init_channels'),
         train_criterion=torch.nn.CrossEntropyLoss,
         data_augmentations=None,
         save_model_str=None, **kwargs):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """


    # instantiate optimize
    
    if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

    gpu = 'cuda:0'
    np.random.seed(seed)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    logging.info('gpu device = %s' % gpu)
    logging.info("config = %s", config)

    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    train_dataset = K49(data_dir, True, data_augmentations)
    test_dataset = K49(data_dir, False, data_augmentations)
    # train_dataset = KMNIST(data_dir, True, data_augmentations)
    # test_dataset = KMNIST(data_dir, False, data_augmentations)
    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    genotype = eval("genotypes.%s" % genotype_class)
    model = Network(init_channels, train_dataset.n_classes, config['n_conv_layers'], genotype)
    model = model.cuda()
    
    total_model_params = np.sum(p.numel() for p in model.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = train_criterion
    criterion = criterion.cuda()
    
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config['initial_lr'], 
                                    momentum=config['sgd_momentum'], 
                                    weight_decay=config['weight_decay'], 
                                    nesterov=config['nesterov'])
    else:
        optimizer = get('opti_dict')[config['optimizer']](model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    
    if config['lr_scheduler'] == 'Cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif config['lr_scheduler'] == 'Exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    logging.info('Generated Network:')
    summary(model, (train_dataset.channels,
                    train_dataset.img_rows,
                    train_dataset.img_cols),
            device='cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        lr_scheduler.step()
        logging.info('epoch %d lr %e', epoch, lr_scheduler.get_lr()[0])
        model.drop_path_prob = config['drop_path_prob'] * epoch / num_epochs

        train_acc, train_obj = train(train_loader, model, criterion, optimizer, grad_clip=config['grad_clip_value'])
        logging.info('train_acc %f', train_acc)

        test_acc, test_obj = infer(test_loader, model, criterion)
        logging.info('test_acc %f', test_acc)


    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)
    
    return test_acc

def create_run_ensemble(model_state_list, 
                        n_layers,
                        grad_clip_value=5, 
                        seed=0, 
                        num_epochs=20,
                        learning_rate=0.001,
                        init_channels=get('init_channels'), 
                        batch_size=get('batch_size'), 
                        genotype_class='PCDARTS'):
    
    if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

    gpu = 'cuda:0'
    np.random.seed(seed)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    logging.info('gpu device = %s' % gpu)
    logging.info("config = %s", config)

    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    train_dataset = K49(data_dir, True, data_augmentations)
    test_dataset = K49(data_dir, False, data_augmentations)
    # train_dataset = KMNIST(data_dir, True, data_augmentations)
    # test_dataset = KMNIST(data_dir, False, data_augmentations)
    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    genotype = eval("genotypes.%s" % genotype_class)
    dataset = dict()
    dims = []
    for i, model_state in enumerate(model_state_list):
        model = Network(init_channels, train_dataset.n_classes, n_layers, genotype)
        model.load_state_dict(torch.load(model_state))
        model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        trn_labels = []
        trn_features = []
        if i == 0:
            for d,la in train_loader:
                o = model(Variable(d.cuda()))
                o = o.view(o.size(0),-1)
                trn_labels.extend(la)
                trn_features.extend(o.cpu().data)
            test_labels = []
            test_features = []
            for d,la in test_loader:
                o = model(Variable(d.cuda()))
                o = o.view(o.size(0),-1)
                test_labels.extend(la)
                test_features.extend(o.cpu().data)
            dataset['trn_labels'] = trn_labels
            dataset['test_labels'] = test_labels

        else:
            for d,la in train_loader:
                o = model(Variable(d.cuda()))
                o = o.view(o.size(0),-1)
                trn_features.extend(o.cpu().data)
            test_labels = []
            test_features = []
            for d,la in test_loader:
                o = model(Variable(d.cuda()))
                o = o.view(o.size(0),-1)
                test_features.extend(o.cpu().data)            
        dataset['trn_features'].extend(trn_features)
        dims.extend(dataset['trn_features'][i][0].size(0))
        dataset['test_features'].extend(test_features)
    

    trn_feat_dset = FeaturesDataset(dataset['trn_features'][0],dataset['trn_features'][1],dataset['trn_features'][2],dataset['trn_labels'])
    test_feat_dset = FeaturesDataset(dataset['test_features'][0],dataset['test_features'][1],dataset['test_features'][2],dataset['test_labels'])
    trn_feat_loader = DataLoader(trn_feat_dset,batch_size=64,shuffle=True)
    test_feat_loader = DataLoader(val_feat_dset,batch_size=64)
    model = EnsembleModel(dims, out_size=train_dataset.n_classes)
    criterion = torch.nn.optim.CrossEntropyLoss
    criterion = criterion.cuda()
    optimizer = torch.nn.optim.SGD(model.parameters(), 
                                    lr=learning_rate, 
                                    momentum=0.9)   
    
    for epoch in range(num_epochs):
        epoch_loss, epoch_accuracy = fit(epoch,model,trn_feat_loader,critierion, training=True)
        val_epoch_loss , val_epoch_accuracy = fit(epoch,model, test_feat_loader, criterion, training=False)


    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if not os.path.exists(save_model_str):
            os.mkdir(save_model_str)
        
        torch.save(model.state_dict(), os.path.join(save_model_str, time.ctime())) 
    
if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS19 final project')

    cmdline_parser.add_argument('-s', '--seed',
                                default=0,
                                help='random seed',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=96,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default='./data',
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(get('opti_dict').keys()),
                                type=str)
    cmdline_parser.add_argument('-n', '--exp_no',
                                default=0,
                                help='Experiment number',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    darts_args = get_darts_args()
    darts_args['seed'] = args.seed
    exp_name = f'EXP{str(args.exp_no)}'
    exp_dir = f"./{exp_name}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    darts(exp_name, darts_args)
    # _ = main(
    #         bohb_config,
    #         data_dir=settings.data_dir,
    #         num_epochs=int(20),
    #         batch_size=bohb_config['batch_size'],
    #         learning_rate=bohb_config['learning_rate'],
    #         train_criterion=settings.loss_dict[bohb_config['training_loss']],
    #         model_optimizer=settings.opti_dict[bohb_config['optimizer']],
    #         data_augmentations=None,  
    #         save_model_str=settings.run_dir
    #         )

