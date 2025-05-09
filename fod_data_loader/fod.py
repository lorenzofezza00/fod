"""FOD Dataloader"""
import os
import numpy as np
import torch
import torch.utils.data as data
import re
from PIL import Image
import pandas as pd
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from fault_injection.WeightFault import WeightFault
from ast import literal_eval as make_tuple

__all__ = ['fod']

class FOD(data.Dataset):
    """Faulty Output Dataset Dataset.

    Parameters
    ----------
    root : string
        Path to FOD folder. Default is ../masks'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = 19

    def __init__(self, root='../faulty_output_dataset', **kwargs):
        super(FOD, self).__init__()
        self.root = root
        self.mask_paths, self.mask_info = _get_fod_pairs(self.root)
        assert (len(self.mask_paths) == len(self.mask_info))
        if len(self.mask_paths) == 0:
            raise RuntimeError("Found 0 masks in subfolders of: " + self.root + "\n")
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
        #                       23, 24, 25, 26, 27, 28, 31, 32, 33]
        # self._key = np.array([-1, -1, -1, -1, -1, -1,
        #                       -1, -1, 0, 1, -1, -1,
        #                       2, 3, 4, -1, -1, -1,
        #                       5, -1, 6, 7, 8, 9,
        #                       10, 11, 12, 13, 14, 15,
        #                       -1, -1, 16, 17, 18])
        # self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    # def _mask_transform(self, mask):
    #     target = self._class_to_index(np.array(mask).astype('int32'))
    #     return torch.LongTensor(np.array(target).astype('int32'))
    
    # def _class_to_index(self, mask):
    #     values = np.unique(mask)
    #     for value in values:
    #         assert (value in self._mapping)
    #     index = np.digitize(mask.ravel(), self._mapping, right=True)
    #     return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        cmask = Image.open(self.mask_paths[index][0])
        cmask = torch.Tensor(np.array(cmask).astype('int32'))
        
        fmask = Image.open(self.mask_paths[index][1])
        fmask = torch.Tensor(np.array(fmask).astype('int32')) # self._mask_transform(mask)
        # Frame,Injection,Layer,TensorIndex,Bit,mIoU,PA,Label
        row = self.mask_info[index]
        frame = row['Frame']
        # fault = WeightFault(
        #         injection=int(row['Injection']),
        #         layer_name=row['Layer'],
        #         tensor_index=make_tuple(row['TensorIndex']),
        #         bit=int(row['Bit'])
        #     )
        fault = {
                'injection':int(row['Injection']),
                'layer_name':row['Layer'],
                'tensor_index':make_tuple(row['TensorIndex']),
                'bit':int(row['Bit'])
            }
        mIoU = float(row['mIoU'])
        pa = float(row['PA'])
        label = float(row['Label'])
        return frame, fault, mIoU, pa, cmask, fmask, label

    def __len__(self):
        return len(self.mask_paths)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS


def _get_fod_pairs(folder):
    def get_path_pairs(mask_folder):
        mask_paths = {}
        mask_info = {}
        df = pd.read_csv(os.path.join(mask_folder ,'faulty_output_dataset.csv'))
        for root, _, files in os.walk(mask_folder):
            for filename in files:
                if filename.endswith(".png"):
                    mask_path = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(mask_path))
                    if foldername == 'c' or foldername == 'nc': 
                        f_maskpath = os.path.join(mask_folder, foldername, filename)
                        match = re.search(r"fm(\d+)_", filename)
                        if match:
                            n = int(match.group(1))
                            info = df.iloc[n]
                            c_maskpath = os.path.join(mask_folder, 'co', f"cm_{info['Frame']}.png")
                        else:
                            print('incorrect mask format:', f_maskpath, c_maskpath)
                        if os.path.isfile(f_maskpath) and os.path.isfile(c_maskpath) and match:
                            mask_paths[n] = (c_maskpath, f_maskpath)
                            mask_info[n] = info
                        else:
                            print('cannot find the mask:', f_maskpath, c_maskpath)

        print('Found {} masks in the folder {}'.format(len(mask_paths), folder))
        return [mask_paths[key] for key in sorted(mask_paths)], [mask_info[key] for key in sorted(mask_info)]
    
    mask_folder = folder
    mask_paths, mask_info = get_path_pairs(mask_folder)
    return mask_paths, mask_info
    
if __name__ == '__main__':
    dataset = FOD()
    frame, fault, mIoU, pa, cmask, fmask, label = dataset[0]
