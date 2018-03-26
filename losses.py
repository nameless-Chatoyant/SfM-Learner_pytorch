import torch
import torch.nn.functional as F

from utils import tolist

def photometric_reconstuction_loss(target, sources, intrinsics, intrinsics_inv, depth, exp_mask, pose, rotation_mode = 'euler', padding_mode = 'zeros'):
    """

    # Arguments
        target:
        sources:
        intrinsics:
        intrinsics_inv:
        depth:
        rotation_mode:
        padding_mode:

    # Returns
        loss
    """
    def one_scale(depth, exp_mask):
        assert(exp_mask is None or depth.size()[2:] == exp_mask.size()[2:])
        assert(pose.size(1) == len(sources))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = target.size(2) / h

        target_scaled = F.adaptive_avg_pool2d(target, (h, w))
        sources_scaled = [F.adaptive_avg_pool2d(i, (h, w)) for i in sources]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv =  torch.cat((intrinsics_inv[:, :, 0:2] * downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for idx, source in enumerate(sources_scaled):
            current_pose = pose[:, i]

            source_warped = inverse_warp(source,
                                        depth[:, 0],
                                        current_pose,
                                        intrinsics_scaled,
                                        intrinsics_scaled_inv,
                                        rotation_mode,
                                        padding_mode)
            out_of_bound = 1 - (source_warped == 0).prod(1, keepdim=True).type_as(source_warped)
            diff = (target_scaled - source_warped) * out_of_bound

            if exp_mask is not None:
                diff = diff * exp_mask[:, idx: idx+1].expand_as(diff)
            
            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).data[0] == 1)
        
        return reconstruction_loss
    

    exp_mask, depth = tolist(exp_mask, depth)
    
    loss = 0
    for d, mask in zip(depth, exp_mask):
        loss += one_scale(d, mask)

    return loss


def exp_loss(exp_mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    
    loss = 0
    for m in exp_mask:
        ones_var = Variable(torch.ones(1)).expand_as(m).type_as(m)
        loss += nn.functional.binary_cross_entropy(m, ones_var)
    return loss


def smooth_loss(disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    
    disp = tolist(disp)

    loss = 0
    weight = 1.

    for disp_scaled in disp:
        dx, dy = gradient(disp_scaled)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)

        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.83 # 2 * 2^0.5

    return loss

