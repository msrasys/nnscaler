# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
For test:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/textnas/dataloader.py
"""


import os
import numpy as np
import torch
from torch.utils import data
import threading
from transformers import BertModel, BertTokenizer
import collections
import time

import cube
from cube.runtime.device import DeviceGroup
from cube.profiler import CudaTimer


def read_sst_2(data_path='./SST-2', max_input_length=64, min_count=1):
    sentences, labels = [], []
    assert os.path.exists(data_path)
    dataset_train = os.path.join(data_path, 'train.tsv')
    with open(dataset_train, 'r') as f:
        lines = f.readlines()[1:] # skip first
    for line in lines:
        sentence, label = line.split('\t')
        sentence = sentence.strip()
        label = int(label.strip())
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels


class SSTDataset(data.Dataset):
    def __init__(self):
        self.sents, self.labels = read_sst_2()
        print(f'> loaded SST dataset: train length: {len(self.sents)}')

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]

    def __len__(self):
        return len(self.sents)


class SharedDataLoader(object):
    def __init__(self, batch_size, replicate=True, **kwargs):
        self.replicate = replicate
        self.has_model = self.replicate or (DeviceGroup().rank == 0)

        dataset = SSTDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, **kwargs)
        self.dataloader = dataloader

        if self.has_model:
            self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.model = None
            self.tokenizer = None

        self.max_queue = 32
        self.input_size = (batch_size, 64, 768)
        self.batch_size = batch_size
        self.length = len(dataset) // batch_size

    def __iter__(self):
        self.counter = 0
        self.shared_queue = collections.deque()
        self._dataloader_iter = iter(self.dataloader)
        if self.has_model and (not self.replicate):
            # sharing mode: all models share the same dataloader
            print('starting pipeline to produce datas')
            self.workers = threading.Thread(target=self._pipe).start()
        return self

    def __len__(self):
        return len(self.dataloader)

    def get_data(self):
        if self.replicate:
            CudaTimer().start('bert')
        text, label = next(self._dataloader_iter)
        text = torch.tensor([self.tokenizer.encode(t, max_length=64, padding='max_length') for t in text]).cuda()
        mask = text > 0
        with torch.no_grad():
            output = self.model(text)['last_hidden_state']
        label = label.cuda()
        if self.replicate:
            CudaTimer().stop('bert')
        return output, mask, label

    def _pipe(self):
        while True:
            while len(self.shared_queue) >= self.max_queue:
                time.sleep(0.2)
            # print('sample data...')
            datas = self.get_data()
            # print(datas)  
            self.shared_queue.append(datas)

    def __next__(self):
        self.counter += 1
        if self.counter >= len(self):
            raise StopIteration
        if self.replicate:
            # replicate mode: each gpu has a dataloader
            text, masks, labels = self.get_data()
        else:
            # sharing mode: all models share the same dataloader
            if self.has_model:
                while not self.shared_queue:
                    time.sleep(0.1)
                text, masks, labels = self.shared_queue.popleft()
                assert torch.is_tensor(text)
                masks = masks.float()
            else:
                text = torch.zeros(self.input_size, dtype=torch.float, device="cuda")
                labels = torch.zeros(self.batch_size, dtype=torch.long, device="cuda")
                masks = torch.zeros(self.input_size[:2], dtype=torch.float, device="cuda")
            CudaTimer().start('get_data')
            torch.distributed.broadcast(text, 0)
            torch.distributed.broadcast(labels, 0)
            torch.distributed.broadcast(masks, 0)
            CudaTimer().stop('get_data')
            masks = masks.bool()
        return text, masks, labels


if __name__ == '__main__':

    cube.init()
    dataloader = SharedDataLoader(32, replicate=True)
    for datas in dataloader:
        print(f'get data: {[data.size() for data in datas]}')
        input('>>>')
