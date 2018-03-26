import torch.nn as nn

from modules import DispNet, PoseExpNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.disp_net = DispNet()
        self.pose_exp_net = PoseExpNet()
    

    def init_weights(self):
        self.disp_net.init_weights()
        self.pose_exp_net.init_weights()


    def forward(self):
        pass



if __name__ == '__main__':
    net = Net()
    print(net)
    
