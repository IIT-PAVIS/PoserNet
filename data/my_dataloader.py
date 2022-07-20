"""
Name: my_dataloader.py
Description: Code adapted from PyTorch Geometric (PyG). We recommend that you use that package directly, for your projects.
             PyG is distributed with an MIT license.
             Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>.
             That license is of the same type as that used for PoserNet, please refer to the LICENSE file for details,
             or look at the LICENSE file for PyG on its GitHub page: https://github.com/pyg-team/pytorch_geometric.
-----
Authors: Matthias Fey, Jiaxuan You, Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from data.my_batch import Batch


class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        return Batch.from_data_list(batch, self.follow_batch,
                                    self.exclude_keys)

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        exclude_keys (list or tuple, optional): Will exclude each key in the
            list. (default: :obj:`[]`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 exclude_keys=[], **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch,
                                                 exclude_keys), **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda data_list: data_list, **kwargs)


class DenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DenseDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=DenseCollater(), **kwargs)
