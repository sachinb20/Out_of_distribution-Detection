##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee 
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import models
import tqdm

from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='..', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--beta', type=float, default=1, help='penalty parameter for KL term')

args = parser.parse_args()

if args.dataset == 'cifar10':
    args.beta = 0.1
    args.batch_size = 64
    
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# print('load data: ',args.dataset)
# train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

# print('Load model')
# model = models.vgg13()
# print(model)



print('load GAN')
nz = 100
netG = models.Generator(1, nz, 64, 3) # ngpu, nz, ngf, nc

print('Load model')
# model = models.vgg13()
netG.load_state_dict(torch.load('/home/dell/OOD_Detection/Confident_classifier/results/joint_confidence_loss/4249/netG_epoch_100.pth'))
print(netG)

fixed_noise = torch.FloatTensor(1, nz, 1, 1).normal_(0, 1)

if args.cuda:
    # model.cuda()
    # netD.cuda()
    netG.cuda()
    # criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)



def generate(epoch):
        netG.eval()
   
        noise = torch.FloatTensor(1, nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        # fake = netG(fixed_noise)
        vutils.save_image(fake.data, '%s/cifar_GAN_OOD/%d.png'%(args.outf, epoch), normalize=True)




for epoch in range(1, args.epochs + 1):
    generate(epoch)
