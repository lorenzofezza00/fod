from .fod import FOD

datasets = {
    'fod': FOD
}


def get_fod_dataset(name, **kwargs):
    """FOD Dataset"""
    return datasets[name.lower()](**kwargs)
