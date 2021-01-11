import argparse
import random
import sys
import os
import json
from time import time

import numpy as np
import torch
import torch.optim as optim
import visdom

# from dataset import *
from data_loader import DataLoader
from model import *
from utils import *
import emd_module as emd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--data_dir', type=str, default='data/shapenet_color',
                        help='Data directory')
    parser.add_argument('--nepoch', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--model', type=str, default='',
                        help='optional reload model path')
    parser.add_argument('--in_num_points', type=int, default=2048,
                        help='number of output points')
    parser.add_argument('--out_num_points', type=int, default=4096,
                        help='number of input points')
    parser.add_argument('--n_primitives', type=int, default=16,
                        help='number of surface elements')
    parser.add_argument('--exp_name', type=str, default="shapenet_02691156",
                        help='Experiment name')
    parser.add_argument('--env', type=str, default="MSN_TRAIN",
                        help='visdom environment')
    parser.add_argument('--category', type=str, default='02691156', metavar='N',
                        help='category of the data')
    parser.add_argument('--isrotate', action='store_true', default=True,
                        help='Whether to rotate the input data along Z axis [default: True]')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaluate the model')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Small dataset to debug')
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Use multiple GPU or not')

    opt = parser.parse_args()
    print(opt)

    return opt


if __name__ == '__main__':
    opt = parse_args()

    class FullModel(nn.Module):
        def __init__(self, model):
            super(FullModel, self).__init__()
            self.model = model
            self.EMD = emd.emdModule()

        def forward(self, inputs, gt, eps, iters):
            output1, output2, expansion_penalty = self.model(inputs)
            gt = gt[:, :, :3]

            dist, _ = self.EMD(output1, gt, eps, iters)
            emd1 = torch.sqrt(dist).mean(1)

            dist, _ = self.EMD(output2, gt, eps, iters)
            emd2 = torch.sqrt(dist).mean(1)

            return output1, output2, emd1, emd2, expansion_penalty

    vis = visdom.Visdom(port=8097, env=opt.env) # set your port
    save_path = opt.exp_name
    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    dir_name =  os.path.join('logs', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    os.system('cp ./main.py %s' % dir_name)
    os.system('cp ./data_loader.py %s' % dir_name)
    os.system('cp ./model.py %s' % dir_name)

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    best_val_loss = 10

    dataset = DataLoader(root=opt.data_dir,
                         npoint=opt.in_num_points,
                         split='train',
                         category=[opt.category],
                         isrotate=opt.isrotate,
                         debug=opt.debug)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=int(opt.workers),
                                            pin_memory=True,
                                            drop_last=True)
    dataset_test = DataLoader(root=opt.data_dir,
                              npoint=opt.in_num_points,
                              split='test',
                              category=[opt.category],
                              isrotate=opt.isrotate,
                              debug=opt.debug)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=int(opt.workers),
                                                  pin_memory=True,
                                                  drop_last=True)

    len_dataset = len(dataset)
    print("Train Set Size: ", len_dataset)

    network = MSN(num_points = opt.out_num_points, n_primitives = opt.n_primitives)
    if opt.multi_gpu:
        network = torch.nn.DataParallel(FullModel(network))
        network.cuda()
        model = network.module.model
    else:
        network = FullModel(network)
        network.cuda()
        model = network.model
    model.apply(weights_init) #initialization of the weight

    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    lrate = 0.001 #learning rate
    optimizer = optim.Adam(model.parameters(), lr = lrate)

    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()
    with open(logname, 'a') as f: #open and append
        f.write(str(model) + '\n')


    train_curve = []
    val_curve = []
    labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.out_num_points//opt.n_primitives)+1)).view(opt.out_num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
    labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
    labels_generated_points = labels_generated_points.contiguous().view(-1)


    for epoch in range(opt.nepoch):
        #TRAIN MODE
        train_loss.reset()
        model.train()

        # learning rate schedule
        if epoch == 20:
            optimizer = optim.Adam(model.parameters(), lr=lrate/10.0)
        if epoch == 40:
            optimizer = optim.Adam(model.parameters(), lr=lrate/100.0)

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            input1, input2, gt1, gt2 = data
            input1 = input1.float().cuda()
            input2 = input2.float().cuda()
            gt1 = gt1.float().cuda()
            gt2 = gt2.float().cuda()
            input1 = input1.transpose(2, 1).contiguous()
            input2 = input2.transpose(2, 1).contiguous()

            output11, output12, emd11, emd12, expansion_penalty1 = \
                network(input1, gt1.contiguous(), 0.005, 50)
            output21, output22, emd21, emd22, expansion_penalty2 = \
                network(input2, gt2.contiguous(), 0.005, 50)
            loss_net1 = emd11.mean() + emd12.mean() + expansion_penalty1.mean() * 0.1
            loss_net2 = emd21.mean() + emd22.mean() + expansion_penalty2.mean() * 0.1
            loss_net_whole = (loss_net1 + loss_net2) / 2
            loss_net_whole.backward()
            train_loss.update((emd12.mean().item() + emd22.mean().item()) / 2)
            optimizer.step()

            if i % 10 == 0:
                idx = random.randint(0, input1.size()[0] - 1)
                vis.scatter(X = gt1.contiguous()[idx].data.cpu()[:, :3],
                        win = 'TRAIN_GT',
                        opts = dict(
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = input1.transpose(2,1).contiguous()[idx].data.cpu(),
                        win = 'TRAIN_INPUT',
                        opts = dict(
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = output11[idx].data.cpu(),
                        Y = labels_generated_points[0:output11.size(1)],
                        win = 'TRAIN_COARSE',
                        opts = dict(
                            markersize=2,
                            ),
                        )
                vis.scatter(X = output12[idx].data.cpu(),
                        win = 'TRAIN_OUTPUT',
                        opts = dict(
                            markersize=2,
                            ),
                        )

            print(opt.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' % (
                epoch, i, len_dataset/opt.batch_size,
                (emd11.mean().item() + emd21.mean().item()) / 2,
                (emd12.mean().item() + emd22.mean().item()) / 2,
                (expansion_penalty1.mean().item() + expansion_penalty2.mean().item()) / 2
                ))
        train_curve.append(train_loss.avg)

        # VALIDATION
        if epoch % 5 == 0:
            val_loss.reset()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(dataloader_test, 0):
                    input1, input2, gt1, gt2 = data
                    input1 = input1.float().cuda()
                    input2 = input2.float().cuda()
                    gt1 = gt1.float().cuda()
                    gt2 = gt2.float().cuda()
                    input1 = input1.transpose(2,1).contiguous()
                    input2 = input2.transpose(2,1).contiguous()

                    output11, output12, emd11, emd12, expansion_penalty1 = \
                        network(input1, gt1.contiguous(), 0.004, 3000)
                    output21, output22, emd21, emd22, expansion_penalty2 = \
                        network(input2, gt2.contiguous(), 0.004, 3000)
                    val_loss.update((emd12.mean().item() + emd22.mean().item()) / 2)
                    idx = random.randint(0, input1.size()[0] - 1)
                    vis.scatter(X = gt1.contiguous()[idx].data.cpu()[:, :3],
                            win = 'VAL_GT',
                            opts = dict(
                                markersize = 2,
                                ),
                            )
                    vis.scatter(X = input1.transpose(2,1).contiguous()[idx].data.cpu(),
                            win = 'VAL_INPUT',
                            opts = dict(
                                markersize = 2,
                                ),
                            )
                    vis.scatter(X = output11[idx].data.cpu(),
                            Y = labels_generated_points[0:output11.size(1)],
                            win = 'VAL_COARSE',
                            opts = dict(
                                markersize=2,
                                ),
                            )
                    vis.scatter(X = output12[idx].data.cpu(),
                            win = 'VAL_OUTPUT',
                            opts = dict(
                                markersize=2,
                                ),
                            )
                    print(opt.env + ' val [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(
                        epoch, i, len_dataset / opt.batch_size,
                        (emd11.mean().item() + emd21.mean().item()) / 2,
                        (emd12.mean().item() + emd22.mean().item()) / 2,
                        (expansion_penalty1.mean().item() + expansion_penalty2.mean().item()) / 2
                        ))

        val_curve.append(val_loss.avg)

        vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)))),
                    Y=np.column_stack((np.array(train_curve),np.array(val_curve))),
                    win='loss',
                    opts=dict(title="emd", legend=["train_curve" + opt.env, "val_curve" + opt.env], markersize=2, ), )
        vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)))),
                    Y=np.log(np.column_stack((np.array(train_curve),np.array(val_curve)))),
                    win='log',
                    opts=dict(title="log_emd", legend=["train_curve"+ opt.env, "val_curve"+ opt.env], markersize=2, ), )

        log_table = {
            "train_loss" : train_loss.avg,
            "val_loss" : val_loss.avg,
            "epoch" : epoch,
            "lr" : lrate,
            "bestval" : best_val_loss,
        }
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(log_table) + '\n')

        print('saving net...')
        torch.save(model.state_dict(), '%s/network.pth' % (dir_name))
