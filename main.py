import numpy as np
import argparse
import importlib
import sys,os
import time
import random
sys.path.append(os.path.abspath('./'))
from chainer import cuda


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='coco', help='dataset to train')
parser.add_argument('--batchsize', type=int, default=2, help='literary')
parser.add_argument('--gpu', type=int, default=1, help='run in  specific GPU')
parser.add_argument('--epochs', type=int, default=501, help='number of epochs to train')
parser.add_argument('--save_every_m', type=int, default=100, help='save the model every n epochs')
parser.add_argument('--save_every_i', type=int, default=1, help='save the image every n epochs')
parser.add_argument('--net', type=str, default='mask_rcnn', help='import the network')
parser.add_argument('--load', nargs=2, type=str, default='', help='loading network parameters')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--fine', dest="fine", action='store_true', help='fine')
parser.add_argument('--size', type=int, default=224, help='size')
parser.add_argument('--path', type=str, default='', help='path to sample data')
parser.add_argument('--stage', type=str, default='rpn', help='rpn/mask')

args = parser.parse_args()
print args

print "==> using network %s" % args.net
args_dict = dict(args._get_kwargs())

network_module = importlib.import_module("nets." + args.net)
network = network_module.Network(**args_dict)


def do_epoch(mode, epoch):
    if mode=='train':
        length=len(network.train_data)
        perm = np.random.permutation(length)
    if mode=='test':
        length=len(network.test_data)
        perm = np.array(range(length))
    sum_loss = np.float32(0)
    sum_accuracy = np.float32(0)
    bs=network.batchsize
    bs2=network.batchsize
    # if mode=='test':
    #     bs2=0
        
    batches_per_epoch=0
    for batch_index in xrange(0, length-bs2, bs):
        batches_per_epoch+=1
        step_data=network.step(perm,batch_index,mode,epoch)
        prediction = step_data["prediction"] 
        current_loss = step_data["current_loss"]
        current_accuracy = step_data["current_accuracy"]

        sum_loss += current_loss
        sum_accuracy += current_accuracy

    if mode=='train':
        print "epoch %d end loss: %.10f"%(epoch, sum_loss/batches_per_epoch),
        print "train accuracy: %.10f"%(sum_accuracy/batches_per_epoch)

    elif mode =='test':
        print 'test loss: %.10f'%(sum_loss/batches_per_epoch),
        print "test accuracy: %.10f"%(sum_accuracy/batches_per_epoch)

start_epoch=0

if args.load != '':
    start_epoch=network.load_state(args.load[0], args.load[1] )

if args.fine:
    start_epoch=0

if args.mode == 'train':
    print "==> training"  
    start=time.time()
    elapsed_time=0
    for epoch in xrange(start_epoch,args.epochs):
        do_epoch('train', epoch)
        if epoch%args.save_every_i==0 :
            do_epoch('test',epoch)
            network.generate(epoch)

        # if epoch % args.save_every_m==0 and epoch != start_epoch:
        #     network.save_params(epoch)

        if epoch%args.save_every_m==0 and epoch != start_epoch:
            network.save_params(epoch)
            print 'changing stage'
            if network.stage=='rpn':
                network.stage='mask'
                network.batchsize=20
            else:
                network.stage='rpn'
                network.batchsize=64
            

elif args.mode == 'test' or args.mode == 'valid':
    print "==> testing"
    network.test()

elif args.mode == 'sample':
    print "==> testing on a given sample"
    network.sample(args.path)