import torch

settings = dict()
settings['bohb_results_dir'] = "./BOHB_results/"
settings['es_results_dir'] = './ES_results'
settings['data_dir'] = './data'
settings['loss_dict'] = {'cross_entropy': torch.nn.CrossEntropyLoss
            }#'mse': torch.nn.MSELoss}
settings['opti_dict'] = {'adam': torch.optim.Adam,
            'adad': torch.optim.Adadelta,
            'sgd': torch.optim.SGD}
settings['run_dir'] = './models'
settings['init_channels'] = 8
settings['batch_size'] = 64
def get(variable):
    if variable in settings.keys():
        return settings[variable]
    else:
        raise KeyError

darts_args=dict()
darts_args['data'] ='./data'    #'location of the data corpus'   
darts_args['set'] ='cifar10'    #'location of the data corpus'   
darts_args['batch_size'] =32    #'batch size'   
darts_args['learning_rate']=0.1    #'init learning rate'   
darts_args['learning_rate_min']=0.001    #'min learning rate'   
darts_args['momentum']=0.9    #'momentum'   
darts_args['weight_decay']=3e-4    #'weight decay'   
darts_args['report_freq']=50    #'report frequency'   
darts_args['gpu']='cuda:0'    #'gpu device id'   
darts_args['epochs']= 20    #'num of training epochs'    
darts_args['init_channels']=8    #'num of init channels'   
darts_args['layers']= 4    #'total number of layers'   
darts_args['model_path']= 'saved_models'    #'path to save the model'   
darts_args['cutout']= False    #'use cutout'   
darts_args['cutout_length']= 16    #'cutout length'   
darts_args['drop_path_prob']= 0.3    #'drop path probability'   
darts_args['save']= 'EXP'    #'experiment name'   
darts_args['seed']=0   #'random seed'   
darts_args['grad_clip']=5    #'gradient clipping'   
darts_args['train_portion']=0.5    #'portion of training data'   
darts_args['unrolled']=False    #'use one-step unrolled validation loss'   
darts_args['arch_learning_rate']=6e-4    #'learning rate for arch encoding'   
darts_args['arch_weight_decay']=1e-3    #'weight decay for arch encoding'   

def get_darts_args():
    return darts_args