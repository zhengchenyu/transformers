# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extraction saving/loading class for common feature extractors.
"""

import copy
from collections.abc import Iterable
from typing import Union

from accelerate import DistributedType, PartialState
from accelerate.data_loader import (
    _PYTORCH_DATALOADER_KWARGS,
    BatchSamplerShard,
    DataLoaderDispatcher,
    DataLoaderShard,
    SkipBatchSampler,
    SkipDataLoader,
)
from accelerate.utils import is_torch_xla_available
from datasets import IterableDataset
from torch.utils.data import BatchSampler, DataLoader, Sampler


if is_torch_xla_available():
    from accelerate.data_loader import MpDeviceLoaderWrapper


class SkipSampler(Sampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], skip_samples=0):
        self.sampler = sampler
        self.skip_samples = skip_samples

    def __iter__(self):
        for i, item in enumerate(self.sampler):
            if i >= self.skip_samples:
                yield item

    def __len__(self):
        return len(self.sampler) - self.skip_samples


def insert_skip_sampler(sampler, skip_samples):
    if isinstance(sampler, BatchSampler):
        new_sampler = copy.deepcopy(sampler)
        if isinstance(sampler, BatchSamplerShard):
            next_sampler = sampler.batch_sampler
            inner_sampler = insert_skip_sampler(next_sampler, skip_samples)
            new_sampler.batch_sampler = inner_sampler
        else:
            next_sampler = sampler.sampler
            inner_sampler = insert_skip_sampler(next_sampler, skip_samples)
            new_sampler.sampler = inner_sampler
        return new_sampler
    else:
        return SkipSampler(sampler, skip_samples=skip_samples)


def skip_first_batches(dataloader, num_batches=0, num_samples=None):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`. Should not be used if
    the original dataloader is a `StatefulDataLoader`.
    """
    state = PartialState()
    if state.distributed_type == DistributedType.XLA:
        device = dataloader.device
        dataloader = dataloader.dataloader

    dataset = dataloader.dataset
    sampler_is_batch_sampler = False
    if isinstance(dataset, IterableDataset):
        assert num_samples is None, "Can't skip samples when the dataset is an IterableDataset."
        new_batch_sampler = None
    elif num_samples is not None:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = insert_skip_sampler(batch_sampler, skip_samples=num_samples)
    else:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = dataloader.batch_size

    if isinstance(dataloader, DataLoaderDispatcher):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        dataloader = DataLoaderDispatcher(
            dataset,
            split_batches=dataloader.split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader._drop_last,
            **kwargs,
        )
    elif isinstance(dataloader, DataLoaderShard):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        elif sampler_is_batch_sampler:
            kwargs["sampler"] = new_batch_sampler
            kwargs["batch_size"] = dataloader.batch_size
        else:
            kwargs["batch_sampler"] = new_batch_sampler
        dataloader = DataLoaderShard(
            dataset,
            device=dataloader.device,
            rng_types=dataloader.rng_types,
            synchronized_generator=dataloader.synchronized_generator,
            **kwargs,
        )
    else:
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            dataloader = SkipDataLoader(dataset, skip_batches=num_batches, **kwargs)
        else:
            dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

    if state.distributed_type == DistributedType.XLA:
        dataloader = MpDeviceLoaderWrapper(dataloader, device)

    return dataloader
