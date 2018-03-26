from easydict import EasyDict as edict

import numpy as np

cfg = edict()


# Camera Parameters
cfg.intrinsics = []

cfg.photo_loss_weight = 1
cfg.mask_loss_weight = 0
cfg.smooth_loss_weight = 0.1

# Datasets
cfg.train_list = []
cfg.test_list = []

cfg.seq_length = 3

# Dataflow
cfg.num_of_workers = 6

# Optimizer
cfg.lr = 2e-4
cfg.momentum = 0.9
cfg.beta = 0.999

