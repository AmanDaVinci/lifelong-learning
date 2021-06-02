import random
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from datasets import Features, Value, ClassLabel
from datastreams.datasets import dataset_configs


class DataStream:
    features = Features({
        "context":      Value("string"),
        "statement":    Value("string"),
        "label":        ClassLabel(2, names=["False", "True"])
    })

    def __init__(self, dataset_names: list, split: str="train_split"):
        self.dataset_names = dataset_names
        self.stream = []
        for name in dataset_names:
            config = dataset_configs[name] 
            path = config["path"]
            name = config.get("name", None)
            dataset_split = config[split]
            dataset = load_dataset(path, name, split=dataset_split)
            filter_column = config.get("filter_column", None)
            filter_value = config.get("filter_value", None)
            if filter_column and filter_value:
                dataset = dataset.filter(lambda batch: batch[filter_column]==filter_value)
            transform = config["transform"]
            dataset = dataset.map(transform, batched=True, remove_columns=dataset.column_names)
            try:
                dataset = dataset.cast(self.features)
            except:
                raise ValueError(f"{transform} didn't transform to datastream features.")
            self.stream.append(dataset)
    
    def summary(self):
        return pd.DataFrame(
            [(name, data.num_rows) for name, data in zip(self.dataset_names, self.stream)],
            columns=["dataset", "num_examples"]
        )
    
    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        for name, data in zip(self.dataset_names, self.stream):
            data.to_pandas().to_csv(path/f"{name}.csv", index=False)

    def sample_examples(self, num_per_dataset: int=1) -> pd.DataFrame:
        all_sample_data = []
        for name, data in zip(self.dataset_names, self.stream):
            sample_idxs = random.choices(range(data.num_rows), k=num_per_dataset)
            sample_data = data.select(sample_idxs).to_pandas()
            sample_data["dataset"] = name
            all_sample_data.append(sample_data)
        return pd.concat(all_sample_data)

    def shuffle_datasets(self, seed: int=None):
        self.stream = [data.shuffle(seed) for data in self.stream]
    
    def limit_datasets(self, max_size: int):
        self.stream = [data.select(range(max_size)) if max_size<=data.num_rows else data
                       for data in self.stream]

    def resize_datasets(self, new_size: int):
        new_stream = []
        for data in self.stream:
            if new_size <= data.num_rows:
                new_stream.append(data.select(range(new_size)))
            elif new_size > data.num_rows:
                size = data.num_rows
                num_duplications, remaining_rows = new_size//size, new_size%size
                # BUG: https://github.com/huggingface/datasets/pull/2025
                # HOTFIX: Create and cache a new dataset using flatten_indices()
                resized_data = [data.flatten_indices()] * num_duplications
                if remaining_rows:
                    resized_data += [data.select(range(remaining_rows)).flatten_indices()] 
                resized_data = concatenate_datasets(resized_data)
                new_stream.append(resized_data)
        self.stream = new_stream
    
    def remix_datasets(self, indices: list):
        assert len(self.stream) == len(indices), \
            "Must have indices for each dataset in the datastream."
        self.stream = [data.select(idxs) if max(idxs)<=data.num_rows else data
                       for data, idxs in zip(self.stream, indices)]
    
    def get_dataloader(self, tokenizer, concatenate: bool, batch_size: int, shuffle_examples: bool):
        tokenizer = partial(tokenizer.batch_encode_plus, 
                            padding="max_length", truncation="longest_first")
        def dataloader(dataset):
            dataset = dataset.map(lambda x: tokenizer(list(zip(x["context"], x["statement"]))),
                                  batched=True, remove_columns=["context", "statement"])
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 
                                                      'attention_mask', 'label'])
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_examples)
        if concatenate:
            # BUG: https://github.com/huggingface/datasets/pull/2025
            # HOTFIX: Create and cache a new dataset using flatten_indices()
            self.stream = [data.flatten_indices() for data in self.stream]
            return dataloader(concatenate_datasets(self.stream))
        else:
            return [dataloader(dataset) for dataset in self.stream]
