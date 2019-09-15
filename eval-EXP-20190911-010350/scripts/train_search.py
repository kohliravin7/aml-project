import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from datasets import KMNIST, K49
from settings import get_darts_args


def darts(exp_name, args):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  args['save'] = './{}/{}-{}'.format(exp_name, args['save'], time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args['save'], scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args['save'], 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  np.random.seed(args['seed'])
  torch.cuda.set_device(args['gpu'])
  cudnn.benchmark = True
  torch.manual_seed(args['seed'])
  cudnn.enabled = True
  torch.cuda.manual_seed(args['seed'])
  logging.info('gpu device = %s' % args['gpu'])
  logging.info("args = %s", args)


  data_augmentations = transforms.ToTensor()
  train_data = KMNIST(args['data'], True, data_augmentations)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args['init_channels'], train_data.n_classes, args['layers'], criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args['learning_rate'],
      momentum=args['momentum'],
      weight_decay=args['weight_decay'])


  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args['train_portion'] * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args['batch_size'],
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args['batch_size'],
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]))

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args['epochs']), eta_min=args['learning_rate_min'])

  architect = Architect(model, args)

  for epoch in range(args['epochs']):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    print(F.softmax(model.betas_normal[2:5], dim=-1))
    #model.drop_path_prob = args['drop_path_prob * epoch / args['epochs
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    if args['epochs']-epoch<=1:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
      utils.save_genotype(genotype, os.path.join(args['save'], 'genotype.json'))

    utils.save(model, os.path.join(args['save'], 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  args = get_darts_args()


  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    if epoch >= 15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args['unrolled'])

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args['grad_clip'])
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args['report_freq'] == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  args = get_darts_args()

  with torch.no_grad():    
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args['report_freq'] == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  args = get_darts_args()
  darts(0, args) 

