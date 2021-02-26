import os
import numpy as np
import random
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from dataset_brats import dataset_brats19
from model import UNet3D
from utils import dice_3D, surface_dice_3D, seed_everything, worker_init_fn

def validate(valid_loader, model, args):
    w, h = 0, args.num_classes
    dice_valid = [[0 for x in range(w)] for y in range(h)]
    wt_dice = []
    surface_dice = [[0 for x in range(w)] for y in range(h)]
    wt_surface_dice = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels,_) in enumerate(valid_loader):
            inputs = Variable(inputs).cuda()
            outputs_orig = model(inputs)
            outputs = outputs_orig.data.max(1)[1].squeeze_(1).cpu().numpy()
            labels = np.array(labels.squeeze(1))
            for bt in range(labels.shape[0]):
                labs = np.unique(labels[bt])
                dsc = dice_3D(np.logical_or(outputs[bt] == 1, outputs[bt] == 3).astype(np.uint8), np.logical_or(labels[bt] == 1, labels[bt] == 3).astype(np.uint8), 1)
                dice_valid[1].append(dsc)
                sd = surface_dice_3D(np.logical_or(outputs[bt] == 1, outputs[bt] == 3).astype(np.uint8), np.logical_or(labels[bt] == 1, labels[bt] == 3).astype(np.uint8), 1, args)
                surface_dice[1].append(sd)
                for cls_idx in range(2, len(labs)):
                    dsc = dice_3D(outputs[bt], labels[bt], labs[cls_idx])
                    dice_valid[cls_idx].append(dsc)
                    sd = surface_dice_3D(outputs[bt], labels[bt], labs[cls_idx], args)
                    surface_dice[cls_idx].append(sd)
                dsc_w = dice_3D((outputs[bt] > 0).astype(np.uint8), (labels[bt] > 0).astype(np.uint8), 1)
                sd_w = surface_dice_3D((outputs[bt] > 0).astype(np.uint8), (labels[bt] > 0).astype(np.uint8), 1, args)
                wt_dice.append(dsc_w)
                wt_surface_dice.append(sd_w)
    return dice_valid, wt_dice, surface_dice, wt_surface_dice

def main():
    parser = argparse.ArgumentParser(description='Brats Training')
    parser.add_argument('--num_classes', default=4, type=int, help="num of classes")
    parser.add_argument('--in_channels', default=4, type=int, help="num of classes")
    parser.add_argument('--batch_size', default=2, type=int,help='mini-batch size')
    parser.add_argument('--lr', default=1e-4, type=float,help='initial learning rate')
    parser.add_argument('--crop_xy', default=192, type=int, help="crop dimension in x and y")
    parser.add_argument('--crop_z', default=128, type=int, help="crop dimension in z")
    parser.add_argument('--init_xy', default=24, type=int, help="initial position of x and y")
    parser.add_argument('--init_z', default=17, type=int, help="initial position of z")
    parser.add_argument('--data_root', default='MICCAI_BraTS_2019_Data_Training', help='data root')
    parser.add_argument('--sd_tolerance', default=[1,2], type=int, help="Surface dice tolerance")
    parser.add_argument('--spacing_mm', default=(1,1,1), type=int, help="Surface dice spacing")
    parser.add_argument('--ckpt_dir', default='ckpt/best_LS', help='data root')
    args = parser.parse_args(args=[])

    dataset_valid = dataset_brats19(args=args, isTrain=False)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=2,
                               worker_init_fn=worker_init_fn)
    
    print('Length of dataset- valid:', dataset_valid.__len__())
    model = UNet3D(in_channels=args.in_channels, out_channels=args.num_classes, isSoftmax=True).cuda()
    model = torch.nn.parallel.DataParallel(model)
    mode_list = ['0.0','0.1' , '0.2', '0.3','_SVLS_181', 'W2']
    for mode in mode_list:
        model.load_state_dict(torch.load(args.ckpt_dir + str(mode) + '.pth.tar'))
        dice_valid, wt_dice, surface_dice, wt_sd = validate(valid_loader, model, args)
        avg_wt_sd = np.array(wt_sd).mean(axis=0)
        avg_wt_std = np.array(wt_sd).std(axis=0)
        avg_dice = []
        std_dice = []
        avg_sd = []
        std_sd = []
        for idx_eval in range(1, len(dice_valid)):
            avg_dice.append(np.mean(dice_valid[idx_eval]))
            std_dice.append(np.std(dice_valid[idx_eval]))
            avg_sd.append(np.array(surface_dice[idx_eval]).mean(axis=0))
            std_sd.append(np.array(surface_dice[idx_eval]).std(axis=0))
        
        print('Smoothing:%s' % mode,'Cases:%d'%len(wt_dice), ' WT: %.3f ± %.3f '%(np.mean(wt_dice),np.std(wt_dice)), ' ET: %.3f ± %.3f '%(avg_dice[2],std_dice[2]),
              ' TC: %.3f ± %.3f '%(avg_dice[0],std_dice[0]))
        for tol_idx in range(len(args.sd_tolerance)):
            print('tolerance: %d  WT: %.3f ± %.3f ET: %.3f ± %.3f TC: %.3f ± %.3f'%(args.sd_tolerance[tol_idx], avg_wt_sd[tol_idx], avg_wt_std[tol_idx], avg_sd[2][tol_idx],
            std_sd[2][tol_idx], avg_sd[0][tol_idx], std_sd[0][tol_idx]))

if __name__ == '__main__':
    seed_everything()
    main()