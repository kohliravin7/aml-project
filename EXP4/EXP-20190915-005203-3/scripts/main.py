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
from model import EnsembleModel, MajorityEnsembleModel
from train import train, infer, ensemble_train, ensemble_infer, majority_predict
from darts_bohamiann import BohamiannWorker
from darts_es import ESWorker
from darts_bohb import run_bohb
from bohb_ensemble import run_bohb_ensemble
import hpbandster.core.result as hpres
from sklearn.ensemble import VotingClassifier
# import hpbandster.visualization as hpvis
import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import BOHB
from datasets import K49, KMNIST, FeaturesDataset
from settings import get, get_darts_args, get_main_args
import utils
import sys
import pickle
import glob
import genotypes
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def run_model(config,
         seed=get('seed'),
         data_dir='./data',
         genotype_class='PCDARTS',
         num_epochs=20,
         batch_size=get('batch_size'),
         init_channels=get('init_channels'),
         train_criterion=torch.nn.CrossEntropyLoss,
         data_augmentations=None,
         save_model_str=None, config_type='BOHB', **kwargs):
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

    criterion = train_criterion()
    criterion = criterion.cuda()
    
    if config['optimizer'] == 'sgd' and config_type=='BOHB':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config['initial_lr'], 
                                    momentum=config['sgd_momentum'], 
                                    weight_decay=config['weight_decay'], 
                                    nesterov=config['nesterov'])
    elif config['optimizer'] == 'sgd' and config_type is not BOHB:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config['initial_lr'], 
                                    momentum=0.9, 
                                    weight_decay=0, 
                                    nesterov=False)
    else:
        optimizer = get('opti_dict')[config['optimizer']](model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    
    if config['lr_scheduler'] == 'Cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif config['lr_scheduler'] == 'Exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # logging.info('Generated Network:')
    # summary(model, (train_dataset.channels,
    #                 train_dataset.img_rows,
    #                 train_dataset.img_cols),
    #         device='cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        logging.info('epoch %d lr %e', epoch, lr_scheduler.get_lr()[0])
        model.drop_path_prob = config['drop_path_prob'] * epoch / num_epochs

        train_acc, train_obj = train(train_loader, model, criterion, optimizer, grad_clip=config['grad_clip_value'])
        logging.info('train_acc %f', train_acc)
        lr_scheduler.step()

        test_acc, test_obj = infer(test_loader, model, criterion)
        logging.info('test_acc %f', test_acc)


    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if not os.path.exists(save_model_str):
            os.mkdir(save_model_str)
        save_model_str += '_'.join(time.ctime())
        save_model_str += '_{}'.format(config_type)

        torch.save(model.state_dict(), save_model_str)

    return test_acc, save_model_str

def create_run_ensemble(model_description, 
                        ensemble_config,
                        seed=get('seed'), 
                        num_epochs=20,
                        data_dir='./data',
                        init_channels=get('init_channels'), 
                        batch_size=get('batch_size'), 
                        genotype_class='PCDARTS',
                        data_augmentations=None,
                        save_model_str=None):

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
    trained_models = []
    for i, model_state in enumerate(model_description.keys()):
        model = Network(init_channels, train_dataset.n_classes, model_description[model_state]['config']['n_conv_layers'], genotype)
        model.load_state_dict(torch.load(model_description[model_state]['model_path']))
        model.cuda()
        model.drop_path_prob = model_description[model_state]['config']['drop_path_prob']
        trained_models.append(model)


    ensemble_model = EnsembleModel(trained_models, dense_units = ensemble_config['dense_units'], out_size=train_dataset.n_classes)
    ensemble_model = ensemble_model.cuda()
    
    summary(ensemble_model, input_size=(1,28,28))
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if ensemble_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=ensemble_config['initial_lr'], 
                                    momentum=ensemble_config['sgd_momentum'], 
                                    weight_decay=ensemble_config['weight_decay'], 
                                    nesterov=ensemble_config['nesterov'])
    else:
        optimizer = get('opti_dict')[ensemble_config['optimizer']](model.parameters(), lr=ensemble_config['initial_lr'], weight_decay=ensemble_config['weight_decay'])
    
    if ensemble_config['lr_scheduler'] == 'Cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif ensemble_config['lr_scheduler'] == 'Exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
   
    print('Started Training')
    for epoch in range(num_epochs):
        logging.info('epoch %d lr %e', epoch, lr_scheduler.get_lr()[0])
        model.drop_path_prob = ensemble_config['drop_path_prob'] * epoch / num_epochs
        for p in ensemble_model.model_1.parameters():
            p.requires_grad=False
        for p in ensemble_model.model_2.parameters():
            p.requires_grad=False
        for p in ensemble_model.model_3.parameters():
            p.requires_grad=False
        for p in ensemble_model.out_classifier.parameters():
            p.requires_grad=True
        train_acc, train_obj, models_avg = ensemble_train(train_loader, ensemble_model, criterion, optimizer, grad_clip=ensemble_config['grad_clip_value'])
        logging.info('train_acc %f', train_acc)
        logging.info('models_avg {}'.format(models_avg))
        lr_scheduler.step()

        test_acc, test_obj, models_avg = ensemble_infer(test_loader, ensemble_model, criterion)
        logging.info('test_acc %f', test_acc)
        logging.info('models_avg {}'.format(models_avg))


    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if not os.path.exists(save_model_str):
            os.mkdir(save_model_str)
        os.path.join(save_model_str, 'ENSEMBLE')
        
        torch.save(ensemble_model.state_dict(), os.path.join(save_model_str, time.ctime())) 

def create_majority_ensemble(model_description, 
                        ensemble_config,
                        seed=get('seed'), 
                        num_epochs=20,
                        data_dir='./data',
                        init_channels=get('init_channels'), 
                        batch_size=get('batch_size'), 
                        genotype_class='PCDARTS',
                        data_augmentations=None,
                        save_model_str=None):

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
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    genotype = eval("genotypes.%s" % genotype_class)
    trained_models = []
    for i, model_state in enumerate(model_description.keys()):
        model = Network(init_channels, train_dataset.n_classes, model_description[model_state]['config']['n_conv_layers'], genotype)
        model.load_state_dict(torch.load(model_description[model_state]['model_path']))
        model.cuda()
        model.drop_path_prob = model_description[model_state]['config']['drop_path_prob']
        trained_models.append(model)


    ensemble_model = MajorityEnsembleModel(trained_models)
    
    print('Started Training')
    for epoch in range(num_epochs):
        test_acc, test_obj, models_avg = majority_predict(test_loader, ensemble_model, criterion, 50)
        logging.info('test_acc %f', test_acc)
        logging.info('models_avg {}'.format(models_avg))


    if save_model_str:
        # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
        if not os.path.exists(save_model_str):
            os.mkdir(save_model_str)
        os.path.join(save_model_str, 'ENSEMBLE')
        
        torch.save(ensemble_model.state_dict(), os.path.join(save_model_str, time.ctime())) 
if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """
    args = get_main_args()
    log_lvl = logging.INFO if args['verbose'] == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    darts_args = get_darts_args()
    exp_no = args['exp_no']
    exp_name = 'EXP{}'.format(exp_no)
    exp_dir = './{}'.format(exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    darts(exp_name, darts_args)
    # model_description = dict()
    # model_description['bohb'] = dict()
    # model_description['bohb']['config'] = {'drop_path_prob': 0.04992452260234029, 'grad_clip_value': 7, 'initial_lr': 0.0028493238730520885, 'lr_scheduler': 'Cosine', 'n_conv_layers': 5, 'optimizer': 'adam', 'weight_decay': 1.465223282456662e-05}#run_bohb('3', iterations=40)
    # _, model_description['bohb']['model_path'] = 'blah', './stored_models/T_h_u_ _S_e_p_ _1_2_ _2_1_:_5_2_:_5_0_ _2_0_1_9_BOHB' #run_model(model_description['bohb']['config'], save_model_str=get('run_dir'), config_type='BOHB')
    
    # model_description['bohamiann'] = dict()
    # worker = BohamiannWorker('./bohamiann', experiment_no=1)
    # model_description['bohamiann']['config'] = {'drop_path_prob': 0.008888888888888894, 'grad_clip_value': 8, 'initial_lr': 0.0012618275345906706, 'lr_scheduler': 'Cosine', 'n_conv_layers': 6, 'optimizer': 'adam', 'weight_decay': 1.0002993808675978e-05} # worker.run_bohamiann(iterations=40)
    # _, model_description['bohamiann']['model_path'] = "blah", './stored_models/F_r_i_ _S_e_p_ _1_3_ _0_1_:_1_1_:_3_2_ _2_0_1_9_BOHAMIANN' #run_model(model_description['bohamiann']['config'], save_model_str=get('run_dir'), config_type='BOHAMIANN')

    # model_description['es'] = dict()
    # es_worker = ESWorker('./es', experiment_no=1)
    # model_description['es']['config'] = {'drop_path_prob': 0.05001692829452364, 'grad_clip_value': 8, 'initial_lr': 0.00018606192447967088, 'lr_scheduler': 'Cosine', 'n_conv_layers': 6, 'optimizer': 'adam', 'weight_decay': 1.0036871953186636e-05} #es_worker.run_es(iterations=40) 
    # _, model_description['es']['model_path'] = 'blah', './stored_models/F_r_i_ _S_e_p_ _1_3_ _0_4_:_0_1_:_0_7_ _2_0_1_9_ES' #run_model(model_description['es']['config'], save_model_str=get('run_dir'), config_type='ES')

    # start = time.time()
    # ensemble_config = {"dense_units": 116, "drop_path_prob": 0.3201740243121082, "grad_clip_value": 7, "initial_lr": 0.04442566012293193, "lr_scheduler": "Cosine", "optimizer": "sgd", "weight_decay": 7.901347741481734e-05, "nesterov": "False", "sgd_momentum": 0.628375546131343} #run_bohb_ensemble(exp_name=1, model_description=model_description, iterations=10)
    # print("time taken ={} s".format(time.time() - start))
    
    # start = time.time()
    # create_majority_ensemble(model_description,
    #                     ensemble_config,
    #                     seed=get('seed'),
    #                     num_epochs=20,
    #                     init_channels=get('init_channels'), 
    #                     batch_size=get('batch_size'), 
    #                     genotype_class='PCDARTS',
    #                     save_model_str=get('run_dir'))
    # print("time taken ={} s".format(time.time() - start))