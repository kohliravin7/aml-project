import torch

global bohb_results_dir 
bohb_results_dir = "../BOHB_results/"
global es_results_dir
es_results_dir = '../ES_results'
global data_dir 
data_dir = '../data'
global loss_dict 
loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss
            }#'mse': torch.nn.MSELoss}
global opti_dict 
opti_dict = {'adam': torch.optim.Adam,
            'adad': torch.optim.Adadelta,
            'sgd': torch.optim.SGD}
run_dir = 'models'
