
from torch.autograd import Variable


from losses import (photometric_reconstuction_loss as photo_criterion,
                    exp_loss as exp_criterion,
                    smooth_loss as smooth_criterion)
from model import Net
from cfgs.config import cfg


def train(model, train_loader, eval_loader)
    model.train()
    
    def one_epoch(epoch):
        for batch_idx, (target_view, source_views, intriinsics, intriinsics_inv) in enumerate(train_loader):

            # ===============================================
            # Input
            # ===============================================
            target_view, intriinsics, intriinsics_inv = Variable(target_view), Variable(intriinsics), Variable(intriinsics_inv)
            source_views = [Varibale(i) for i in source_views]


            # ===============================================
            # Forward
            # ===============================================
            disp = model.disp_net(target_view)
            depth = [1/i for i in disp]
            exp_mask, pose = model.pose_exp_net(target_view, source_views)


            # ===============================================
            # Loss Function
            # ===============================================
            # Compute Losses
            photo_loss = photo_criterion(target_view,
                                        source_views,
                                        intriinsics,
                                        intrinsics_inv,
                                        depth,
                                        exp_mask,
                                        pose,
                                        rotation_mode='euler',
                                        padding_mode='zeros' )
            exp_loss = exp_criterion(exp_mask)
            smooth_loss = smooth_criterion(disp)

            loss = cfg.photo_loss_weight  *  photo_loss +
                cfg.mask_loss_weight   *  exp_loss +
                cfg.smooth_loss_weight *  smooth_loss

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        

            # ===============================================
            # Summary
            # ===============================================
    
    for i in range(cfg.max_epoches):
        one_epoch(i)


if __name__ == '__main__':
    pass