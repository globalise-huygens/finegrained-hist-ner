from collections import Counter
from datasets import Dataset, DatasetDict
import json
import lightning as L
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def create_dataset(dataset, tagset):
    """Create HuggingFace dataset from jsonl data, converting ner labels to their tagset index."""
    datasets = {}
    with open(dataset) as f:
        d = json.load(f)
    num_classes = len(tagset)
    for split in d:
        dic = {"tokens": d[split]["tokens"]}
        if "labels" in d[split]:
            dic["labels"] = [[tagset[x] for x in seq] for seq in d[split]["labels"]]
        if "unk_classes" in d[split]:
            unks = []
            for unk in d[split]["unk_classes"]:
                if unk:
                    m = np.zeros(num_classes, dtype=bool)
                    m[unk] = True
                    unks.append(m)
                else:
                    unks.append(np.zeros(num_classes, dtype=bool))
            dic["unk_classes"] = unks

        datasets[split] = Dataset.from_dict(dic)
        # copy validation data to predict data to get predictions for validation data
        if split == "validation":
            datasets["predict"] = Dataset.from_dict(dic)

    return DatasetDict(datasets)


def map_begin_to_mid_label(tagset):
    """map indices of B labels to I labels"""
    vals = list(tagset.values())
    for k, v in tagset.items():
        if k.startswith("B-") and k.replace("B-", "I-") in tagset:
            vals[v] = tagset[k.replace("B-", "I-")]
    return vals


def token_start_aligner(tokenizer):
    def align_tokens(instances):
        encoding = tokenizer(
            instances["tokens"], is_split_into_words=True, padding="longest"
        )
        encoding["token_mask"] = get_token_starts(instances, encoding)
        return encoding

    return align_tokens


def get_token_starts(batch, encoding):
    word_id = None
    token_start_masks = []
    for i, _ in enumerate(batch["tokens"]):
        token_start_mask = []
        for wid in encoding.word_ids(batch_index=i):
            if wid is None:
                token_start_mask.append(False)
            elif wid != word_id:
                word_id = wid
                token_start_mask.append(True)
            else:
                token_start_mask.append(False)
        token_start_masks.append(token_start_mask)
    return token_start_masks


def get_subtoken_labels(batch, encoding, b2i):
    label_batches = batch["labels"]
    aligned_labels_batches = []
    for i, labels in enumerate(label_batches):
        aligned_labels = []
        word_id = None
        for wid in encoding.word_ids(batch_index=i):
            if wid is None:
                aligned_labels.append(-100)
            elif wid != word_id:
                aligned_labels.append(labels[wid])
                word_id = wid
            else:
                aligned_labels.append(b2i[aligned_labels[-1]])
        aligned_labels_batches.append(aligned_labels)
    return aligned_labels_batches


def token_start_and_label_aligner(tokenizer, b2i):
    def align_tokens(instances):
        encoding = tokenizer(
            instances["tokens"], is_split_into_words=True, padding="longest"
        )
        encoding["token_mask"] = get_token_starts(instances, encoding)
        if "labels" in instances:
            encoding["labels"] = get_subtoken_labels(instances, encoding, b2i)
        return encoding

    return align_tokens


def token_label_aligner(tokenizer, b2i):
    def align_tokens(instances):
        """split tokens with tokenizer and assign label to each.

        the O label is 0, B labels are uneven and I labels are the following
        even figure. Non initial subtokens receive the same label
        as the initial subtoken except if that is a B label (convert to I label).
        """
        encoding = tokenizer(
            instances["tokens"], padding=True, is_split_into_words=True
        )
        if "labels" in instances:
            encoding["labels"] = get_subtoken_labels(instances, encoding, b2i)
        return encoding

    return align_tokens


def class_weights(data, max_weight):
    c = Counter()
    for x in data:
        c.update(Counter(x["labels"]))
    del c[-100]
    maximum = max(c.values())
    c2 = {k: min(maximum / v, max_weight) for k, v in c.items()}
    return [x[1] for x in sorted(c2.items(), key=lambda x: x[0])]


class NerDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size=32,
        num_workers=1,
        pretrained_model="xlm-roberta-base",
        tagset_path="resources/tagsets/ner_tagset.json",
        compute_class_weights=False,
        max_weight=1000,
        class_weights_path="class_weights.json",
        data_pkl="data.pkl",
        use_unk_classes=False,
        predict_with_labels=True,
        predict_key="predict",
    ):
        super().__init__()
        with open(tagset_path) as f:
            self.tagset = json.load(f)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pkl = data_pkl
        self.num_labels = len(self.tagset)
        self.b2i = map_begin_to_mid_label(self.tagset)
        self.align_subtoken_labels = token_label_aligner(
            AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True),
            self.b2i,
        )
        self.align_token_starts_and_subtoken_labels = token_start_and_label_aligner(
            AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True),
            self.b2i,
        )
        self.align_token_starts = token_start_aligner(
            AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True),
        )
        self.compute_class_weights = compute_class_weights
        self.max_weight = max_weight
        self.class_weights_path = class_weights_path
        self.pretrained_model = pretrained_model
        self.predict_with_labels = predict_with_labels  # for seqeval prediction writer
        self.predict_key = predict_key
        self.datacols = ["input_ids", "attention_mask", "labels"]
        if use_unk_classes:
            self.datacols.append("unk_classes")

    def prepare_data(self):
        if not os.path.exists(self.data_pkl):
            dataset = create_dataset(self.dataset, self.tagset)
            tokenized = {}
            if "train" in dataset:
                tokenized["train"] = dataset["train"].map(
                    self.align_subtoken_labels, batched=True, batch_size=None
                )
            if "validation" in dataset:
                tokenized["validation"] = dataset["validation"].map(
                    self.align_token_starts_and_subtoken_labels,
                    batched=True,
                    batch_size=None,
                )
            if "test" in dataset:
                tokenized["test"] = dataset["test"].map(
                    self.align_token_starts_and_subtoken_labels,
                    batched=True,
                    batch_size=None,
                )
            if self.predict_key in dataset:
                tokenized["predict"] = dataset[self.predict_key].map(
                    self.align_token_starts_and_subtoken_labels,
                    batched=True,
                    batch_size=None,
                )

            if self.compute_class_weights:
                with open(
                    self.class_weights_path,
                    "w",
                ) as f:
                    json.dump(class_weights(tokenized["train"], self.max_weight), f)
            with open(self.data_pkl, "wb") as f:
                pickle.dump(tokenized, f)

    def setup(self, stage: str = "fit"):
        with open(self.data_pkl, "rb") as f:
            tokenized = pickle.load(f)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            tokenized["train"].set_format(type="torch", columns=self.datacols)
            tokenized["validation"].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels", "token_mask"],
            )
            self.train_data = NerDataset(tokenized)
            self.val_data = NerDataset(tokenized, "validation")

        if stage == "test":
            tokenized["test"].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels", "token_mask"],
            )
            self.test_data = NerDataset(tokenized, "test")

        if stage == "predict":
            predict_cols = ["input_ids", "attention_mask", "token_mask"]
            if self.predict_with_labels:
                predict_cols.append("labels")
            tokenized["predict"].set_format(type="torch", columns=predict_cols)
            self.predict_data = NerDataset(tokenized, "predict")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


class NerDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows
