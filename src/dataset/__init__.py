import _dataset


def set_datamodel(datamodel_module):
    _dataset._all_classes = datamodel_module.all_classes
    
Dataset = _dataset.Dataset