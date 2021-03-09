import matplotlib.pyplot as plt
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
from utils import dice_3D, seed_everything, worker_init_fn, EDiceLoss
from calibration_metrics import ece_eval, tace_eval, reliability_diagram
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def step_valid(data_loader, model, metric):
    ece_all, acc_all, conf_all, Bm_all, tace_all = [], [], [], [], []
    losses, metrics, metrics_sd = [], [], []
    model.eval()
    for i, batch in enumerate(data_loader):
        inputs, targets,_ = batch
        targets = targets.squeeze().cuda(non_blocking=True)
        inputs = inputs.float().cuda()
        segs = model(inputs)
        outputs = F.softmax(segs, dim=1).detach().cpu().numpy()
        if len(targets.shape) < 4:#if batch size=1
            targets = targets.unsqueeze(0)
        labels = targets.detach().cpu().numpy()
        
        ece, acc, conf, Bm = ece_eval(outputs,labels)
        tace, _, _, _ = tace_eval(outputs,labels)
        ece_all.append(ece)
        acc_all.append(acc)
        conf_all.append(conf)
        Bm_all.append(Bm)
        tace_all.append(tace)
        segs = segs.data.max(1)[1].squeeze_(1)
        metric_ = metric.metric_brats(segs, targets)
        metrics_sd.extend(metric.get_surface_dice(segs.detach().cpu().numpy(), targets.detach().cpu().numpy()))
        metrics.extend(metric_)

    ece_avg = np.stack(ece_all).mean(0)
    acc_avg = np.stack(acc_all).mean(0)
    conf_avg = np.stack(conf_all).mean(0)
    Bm_avg = np.stack(Bm_all).mean(0)
    tace_avg = np.stack(tace_all).mean(0)
    return metrics, metrics_sd, ece_avg, acc_avg, conf_avg, Bm_avg, tace_avg

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
    parser.add_argument('--ckpt_dir', default='ckpt_brats19/', help='data root')
    args = parser.parse_args()

    dataset_valid = dataset_brats19(args=args, isTrain=False)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=2,
                               worker_init_fn=worker_init_fn)
    
    print('Length of dataset- valid:', dataset_valid.__len__())
    model = UNet3D(in_channels=args.in_channels, out_channels=args.num_classes, isSoftmax=False).to(device)
    model = torch.nn.parallel.DataParallel(model)
    criterion_dice = EDiceLoss().to(device)
    metric = criterion_dice.metric_brats
    legends = ['OH', 'LS(0.1)', 'LS(0.2)', 'LS(0.3)', 'SVLS']
    model_list = ['best_oh.pth.tar', 'best_ls0.1.pth.tar', 'best_ls0.2.pth.tar', 'best_ls0.3.pth.tar', 'best_svls.pth.tar']
    for model_name, legend in zip(model_list, legends):
        model.load_state_dict(torch.load(args.ckpt_dir + str(model_name)))
        model.eval()
        with torch.no_grad():
            dice_metrics, metrics_sd, ece_avg, acc_avg, conf_avg, Bm_avg, tace_avg = step_valid(valid_loader, model, criterion_dice)
        if legend != 'LS(0.3)':
            reliability_diagram(conf_avg, acc_avg, legend=legend)
        dice_metrics = list(zip(*dice_metrics))
        dice_metrics = [torch.tensor(dice, device="cpu").numpy() for dice in dice_metrics]
        avg_dices = np.mean(dice_metrics,1)
        avg_std = np.std(dice_metrics,1)

        metrics_sd = list(zip(*metrics_sd))
        metrics_sd = [torch.tensor(dice, device="cpu").numpy() for dice in metrics_sd]
        avg_sd = np.mean(metrics_sd,1)
        avg_std_sd = np.std(metrics_sd,1)

        print('model:%s , dice[ET:%.3f ± %.3f, TC:%.3f ± %.3f, WT:%.3f ± %.3f], ECE:%.4f, TACE:%.4f'%(
            model_name, avg_dices[0],avg_std[0], avg_dices[1],avg_std[1], avg_dices[2],avg_std[2], ece_avg, tace_avg))
        
        print('model:%s , Surface dice[ET:%.3f ± %.3f, TC:%.3f ± %.3f, WT:%.3f ± %.3f]'%(
            model_name, avg_sd[0],avg_std_sd[0], avg_sd[1],avg_std_sd[1], avg_sd[2],avg_std_sd[2]))

if __name__ == '__main__':
    seed_everything()
    main()