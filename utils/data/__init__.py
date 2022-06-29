def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'olivia':
        from .olivia_data import OliviaDataset
        return OliviaDataset
    elif dataset_name == 'testclothes':
        from .testset_clothes import TestsetClothesDataset
        return TestsetClothesDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
