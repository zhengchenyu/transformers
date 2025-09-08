# Copyright 2018 the HuggingFace Inc. team.
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

from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from accelerate.test_utils.testing import AccelerateTestCase
from torch.utils.data import BatchSampler

from transformers.data_loader import skip_first_batches


class DataLoaderTester(AccelerateTestCase):
    def test_skip_first_batches_by_samples(self):
        dataset = list(range(16))
        # 1 Read samples by two processes
        ## 1.1 Read samples by 0 index
        batch_sampler = BatchSampler(dataset, batch_size=4, drop_last=False)
        batch_sampler_0 = BatchSamplerShard(batch_sampler=batch_sampler, num_processes=2, process_index=0)
        shard_dataloader_0 = DataLoaderShard(dataset=dataset, batch_sampler=batch_sampler_0)
        shard_dataloader_0 = skip_first_batches(shard_dataloader_0, num_samples=0)
        shard_dataloader_iter_0 = iter(shard_dataloader_0)
        assert next(shard_dataloader_iter_0).tolist() == [0, 1, 2, 3]
        ## 1.2 Read samples by 1 index
        batch_sampler = BatchSampler(dataset, batch_size=4, drop_last=False)
        batch_sampler_1 = BatchSamplerShard(batch_sampler=batch_sampler, num_processes=2, process_index=1)
        shard_dataloader_1 = DataLoaderShard(dataset=dataset, batch_sampler=batch_sampler_1)
        shard_dataloader_1 = skip_first_batches(shard_dataloader_1, num_samples=0)
        shard_dataloader_iter_1 = iter(shard_dataloader_1)
        assert next(shard_dataloader_iter_1).tolist() == [4, 5, 6, 7]

        # 2 Continue to read samples.
        # 2.1 Read samples by 0 index with skip
        batch_sampler = BatchSampler(dataset, batch_size=4, drop_last=False)
        batch_sampler_0 = BatchSamplerShard(batch_sampler=batch_sampler, num_processes=3, process_index=0)
        shard_dataloader_0 = DataLoaderShard(dataset=dataset, batch_sampler=batch_sampler_0)
        shard_dataloader_0 = skip_first_batches(shard_dataloader_0, num_samples=8)
        shard_dataloader_iter_0 = iter(shard_dataloader_0)
        assert next(shard_dataloader_iter_0).tolist() == [8, 9, 10, 11]

        # 2.2 Read samples by 1 index with skip
        batch_sampler = BatchSampler(dataset, batch_size=4, drop_last=False)
        batch_sampler_1 = BatchSamplerShard(batch_sampler=batch_sampler, num_processes=3, process_index=1)
        shard_dataloader_1 = DataLoaderShard(dataset=dataset, batch_sampler=batch_sampler_1)
        shard_dataloader_1 = skip_first_batches(shard_dataloader_1, num_samples=8)
        shard_dataloader_iter_1 = iter(shard_dataloader_1)
        assert next(shard_dataloader_iter_1).tolist() == [12, 13, 14, 15]

        # 2.3 Read samples by 2 index with skip
        batch_sampler = BatchSampler(dataset, batch_size=4, drop_last=False)
        batch_sampler_2 = BatchSamplerShard(batch_sampler=batch_sampler, num_processes=3, process_index=2)
        shard_dataloader_2 = DataLoaderShard(dataset=dataset, batch_sampler=batch_sampler_2)
        shard_dataloader_2 = skip_first_batches(shard_dataloader_2, num_samples=8)
        shard_dataloader_iter_2 = iter(shard_dataloader_2)
        assert next(shard_dataloader_iter_2).tolist() == [8, 9, 10, 11]
