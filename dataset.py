import torch.utils.data as data_utils


def load_calib(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


class Data(data_utils.Dataset):

    def __init__(self, files, transformer, seed, seq_length):
        pass
    
    def __getitem__(self, idx):
        pass
    

    def __len__(self):
        pass


if __name__ == '__main__':
    pass