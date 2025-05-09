"""Visualization Utils"""
from PIL import Image
import csv
from ast import literal_eval as make_tuple

__all__ = ['get_color_pallete', 'load_faults', 'filter_faults', 'cityscapes_labels_dict']

def get_color_pallete(npimg):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(cityspallete)
    return out_img

def load_faults(fault_list_path):
    with open(fault_list_path, newline='') as f_list:
        reader = csv.reader(f_list)

        fault_list = list(reader)[1:]

        fault_list = {
            int(fault[0]):
            # WeightFault(
            #     injection=int(fault[0]),
            #     layer_name=fault[1],
            #     tensor_index=make_tuple(fault[2]),
            #     bit=int(fault[-1])
            # )
            {
                'injection':int(fault[0]),
                'layer_name':fault[1],
                'tensor_index':make_tuple(fault[2]),
                'bit':int(fault[-1])
            }
            for fault in fault_list
        }
    
    return fault_list

def filter_faults(fault_list, **filters):
    def matches(fault):
        for attr, value in filters.items():
            if isinstance(fault, dict):
                if fault.get(attr) != value:
                    return False
            else:
                if not hasattr(fault, attr) or getattr(fault, attr) != value:
                    return False
        return True

    return {
        key: fault
        for key, fault in fault_list.items()
        if matches(fault)
    }

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


cityscapes_labels_dict = {
    0: {
        "name": "road",
        "color": (128, 64, 128),
      },
    1: {
        "name": "sidewalk",
        "color": (244, 35, 232),
      },
    2: {
        "name": "building",
        "color": (70, 70, 70),
      },
    3: {
        "name": "wall",
        "color": (102, 102, 156),
      },
    4: {
        "name": "fence",
        "color": (190, 153, 153),
      },
    5: {
        "name": "pole",
        "color": (153, 153, 153),
      },
    6: {
        "name": "traffic light",
        "color": (250, 170, 30),
      },
    7: {
        "name": "traffic sign",
        "color": (220, 220, 0),
      },
    8: {
        "name": "vegetation",
        "color": (107, 142, 35),
      },
    9: {
        "name": "terrain",
        "color": (152, 251, 152),
      },
    10: {
        "name": "sky",
        "color": (70, 130, 180),
      },
    11: {
        "name": "person",
        "color": (220, 20, 60),
      },
    12: {
        "name": "rider",
        "color": (255, 0, 0),
      },
    13: {
        "name": "car",
        "color": (0, 0, 142),
      },
    14: {
        "name": "truck",
        "color": (0, 0, 70),
      },
    15: {
        "name": "bus",
        "color": (0, 60, 100),
      },
    16: {
        "name": "train",
        "color": (0, 80, 100),
      },
    17: {
        "name": "motorcycle",
        "color": (0, 0, 230),
      },
    18: {
        "name": "bicycle",
        "color": (119, 11, 32),
      }
}