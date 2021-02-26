import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import surface_distance
import matplotlib.pyplot as plt

def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 2 ** 32 - 1))

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dice_3D(im1, im2, tid):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def surface_dice_3D(im1, im2, tid, args):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # compute surface dice
    surface_dist = surface_distance.compute_surface_distances(
        im1, im2, spacing_mm=args.spacing_mm)
    sd = []
    for tol_weight in args.sd_tolerance:
        if len(np.unique(im1)) == 1:
            sd_gt2pred = 0
        else:
            sd_gt2pred, _ = surface_distance.compute_surface_overlap_at_tolerance(surface_dist, tol_weight)
        sd.append(sd_gt2pred)
    return sd