from torch.nn import DataParallel
from torch.utils.data import DataLoader
from model import Net
train_loader = 0
val_loader = 0
pass

from dataset import Data
import transformer
def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')

    args = parser.parse_args()

    return args

def main(args):
    # ===============================================
    # Build Data Loader
    # ===============================================
    train_transformer = transformer.Compose([
        transformer.RandomHorizontalFlip(),
        transformer.RandomScaleCrop(),
        transformer.ArrayToTensor(),
        transformer.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = Data('train',
                        transformer = train_transformer
                        seed = args.seed,
                        train = True,
                        seq_length = 3)
    valid_transformer = transformer.Compose([transformer.ArrayToTensor(), normalize])
    val_dataset = Data('val',
                        transformer = valid_transformer
                        seed = args.seed,
                        train = True,
                        seq_length = 3)
    train_loader = DataLoader(train_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = cfg.workers,
                            pin_memory = True)
    val_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = cfg.workers,
                            pin_memory = True)


    # ===============================================
    # Build Model
    # ===============================================
    model = Net()
    model = model.cuda()
    if args.load:
        pass
    else:
        model.init_weights()
    
    model = DataParallel(model)


    # ===============================================
    # Train & Eval at intervals
    # ===============================================



if __name__ == '__main__':
    args = parse()