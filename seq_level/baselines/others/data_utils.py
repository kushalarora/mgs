import torch
import pickle
from torch.utils.data import Dataset


def load_dataset(pad_idx, args):
    dataset_dict = pickle.load(open(args.data_path, 'rb'))
    dataset = {}
    for mode, data in dataset_dict.items():
        chunk_size = args.chunk_size_train if mode == 'train' else args.chunk_size_valid
        dataset[mode] = Corpus(data=data,
                               chunk_size=chunk_size,
                               token_limit_train=args.token_limit_train,
                               token_limit_valid=args.token_limit_valid,
                               context_length=args.context_length,
                               pad_idx=pad_idx,
                               mode=mode)
    return dataset


class Corpus(Dataset):
    def __init__(
            self,
            data,
            chunk_size,
            token_limit_train,
            token_limit_valid,
            context_length,
            pad_idx,
            mode
    ):
        self.chunk_size = chunk_size
        self.pad_idx = pad_idx

        original_length = len(data)

        data = [d for d in data if len(d) > context_length]

        if mode == 'train':
            if token_limit_train > 0:
                data = [d for d in data if len(d) <= token_limit_train]
        else:
            if token_limit_valid > 0:
                data = [d for d in data if len(d) <= token_limit_valid]

        self.batches = self._make_batches(data)

        print('%s size: %d (%d discarded) (max_length %d) (%d batches).'
              % (mode, len(data), original_length - len(data), max(len(d) for d in data), len(self.batches)))

    def _make_batches(self, data):
        sorted_data = sorted(data, key=lambda x: -len(x))
        batches = []
        i = 0
        while i < len(sorted_data):
            sentence = sorted_data[i]
            length = len(sentence)
            batch_size = max(1, self.chunk_size // length)
            batch = sorted_data[i:i + batch_size]
            batch = self._pad_batch(batch, length)
            batches.append(batch)
            i = i + batch_size
        return batches

    def _pad_batch(self, batch, max_length):
        batch_ = []
        for sentence in batch:
            sentence_ = sentence + [self.pad_idx] * (max_length - len(sentence))
            batch_.append(sentence_)
        return batch_

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index], dtype=torch.long)
