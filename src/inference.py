from cassis import load_typesystem, load_cas_from_xmi
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
import lightning as L
import logging
import math
import numpy as np
import operator
import os
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
import zipfile as z

from datamodules import get_token_starts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
NAMED_ENTITY = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ZipInvnrDataset(IterableDataset):
    """Iterates over inventory-numbers Zip files.

    Each zip file contains the XMI pages for that inventory number.
    Workers are assigned a range of zipfiles, which are preordered by size.

    Args:
        inputdir (str): path to zip files
        typesystem (str): path to TypeSystem for XMI reading
        num_workers: number of workers
        max_nb_tokens: max sequence length for inference
    """

    def __init__(self, inputdir, typesystem, num_workers, max_nb_tokens):
        self.inputdir = inputdir
        self.num_workers = num_workers
        self.max_nb_tokens = int(max_nb_tokens)
        with open(typesystem, "rb") as f:
            self.typesystem = load_typesystem(f)

        self.pattern = re.compile(" +")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.num_workers = 1
        self.zipfiles = bucket_files_by_size(inputdir, num_workers)

    def _gen_sequences(self, files):
        for zipfile in files:
            zip_handle = z.ZipFile(zipfile)
            zipmembers = [
                x.filename
                for x in zip_handle.infolist()
                if not x.is_dir() and x.file_size > 0
            ]
            for doc in zipmembers:
                with zip_handle.open(doc, "r") as f:
                    sequences, token_offsets = cas_sequences(
                        f, self.typesystem, self.max_nb_tokens
                    )
                    yield {
                        "tokens": sequences,
                        "offsets": token_offsets,
                        "doc": doc,
                        "invnr": str(zipfile),
                    }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
        else:
            worker_id = worker_info.id
        return self._gen_sequences(self.zipfiles[worker_id])

    def __del__(self):
        """Clean all file handles of the workers on exit

        https://github.com/sara-nl/MLonHPC_May2023/blob/main/Packed_Data_Format/notebooks/packed_data_formats-ANSWERS.ipynb
        """
        if hasattr(self, "zip_handle"):
            for o in self.zip_handle.values():
                o.close()

    def __getstate__(self):
        """Serialize without the ZipFile references, for multiprocessing compatibility

        https://github.com/sara-nl/MLonHPC_May2023/blob/main/Packed_Data_Format/notebooks/packed_data_formats-ANSWERS.ipynb
        """
        state = dict(self.__dict__)
        state["zip_handle"] = {}
        return state


def bucket_files_by_size(dirpath, nbbuckets=1):
    """returns list of lists"""
    logger.info(f"Bucketing zip files in {dirpath}...")
    logger.info(f"Path(dirpath): {Path(dirpath)}...")
    if nbbuckets == 1:
        result = [list(Path(dirpath).glob("**/*.zip"))]
        logger.info(f"Bucketing {len(result[0])} files into a single bucket")
        return result
    zipfiles = [(f, os.path.getsize(f))
                for f in Path(dirpath).glob("**/*.zip")]
    zipfiles.sort(key=operator.itemgetter(1), reverse=True)
    # bucket so as to reach sublists of comparable sizes
    buckets = [[[f[0]], f[1]] for f in zipfiles[:nbbuckets]]
    for f in zipfiles[nbbuckets:]:
        buckets[-1][0].append(f[0])
        buckets[-1][1] += f[1]
        if buckets[-2][1] < buckets[-1][1]:
            buckets.sort(key=operator.itemgetter(1), reverse=True)
    result = [f[0] for f in buckets]
    logger.info(f"Bucketing {len(zipfiles)} files into {len(result)} buckets")
    if len(result) < nbbuckets:
        result.extend([[]] * (nbbuckets - len(result)))
        logger.info(f"Extending buckets to reach size {nbbuckets}")
    return result


def cas_sequences(f, typesystem, max_nb_tokens):
    """Returns token sequences (one sequence per Sentence chunk) and token offsets"""
    cas = load_cas_from_xmi(f, typesystem=typesystem)
    seqs = []
    token_offsets = []
    tokens = cas.select(TOKEN)
    chunked_tokens = split_sequences(
        [t.get_covered_text() for t in tokens],
        max_nb_tokens,
    )
    seqs.extend(chunked_tokens)
    # return token_offsets in a single flat list for a given document
    token_offsets.extend([t.begin, t.end] for t in tokens)
    return seqs, token_offsets


def split_sequences(tokens, max_nb_tokens):
    dest_tokens = []
    while len(tokens) > max_nb_tokens:
        dest_tokens.append(tokens[:max_nb_tokens])
        tokens = tokens[max_nb_tokens:]

    dest_tokens.append(tokens)
    return dest_tokens


class InvnrDataModule(L.LightningDataModule):
    def __init__(
        self,
        inputdir,
        typesystem,
        tagset,
        max_nb_tokens=240,
        num_workers=16,
        pretrained_model="globalise/globalise-NER",
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.tagset_path = tagset
        self.typesystem = typesystem
        self.inputdir = inputdir
        self.max_nb_tokens = int(max_nb_tokens)

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            self.predict_data = ZipInvnrDataset(
                self.inputdir,
                self.typesystem,
                self.num_workers,
                self.max_nb_tokens,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=None,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def collate_fn(self, instance):
        sample = self.tokenizer(
            instance["tokens"],
            is_split_into_words=True,
            padding="longest",
            return_tensors="pt",
        )
        if sample["input_ids"].size()[1] >= self.tokenizer.model_max_length:
            sample = self.tokenizer(
                instance["tokens"],
                is_split_into_words=True,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            print(
                f"WARNING: instance has more subtokens than allowed by model ({sample['input_ids'].size()[1]}) and will be truncated; you should probably lower the maximum token length in the dataset."
            )
        sample["token_mask"] = torch.tensor(get_token_starts(instance, sample))
        sample["offsets"] = torch.tensor(instance["offsets"])
        sample["doc"] = torch.tensor(self.tokenizer.encode(instance["doc"]))
        sample["invnr"] = torch.tensor(
            self.tokenizer.encode(instance["invnr"]))
        return sample


class InvnrWriter(BasePredictionWriter):
    def __init__(self, outdir="."):
        super().__init__("batch")
        self.outdir = outdir

        #  0 maps to 0, B labels are uneven, I labels are even
        # x marks an entity start iff x maps to B label, or else to a I label while the previous token does not map to a B or I label
        # of the same type
        def starts_entity(tag_id, previous_tag_id):
            return (
                tag_id % 2 == 1
                or tag_id > 0
                and tag_id != previous_tag_id + 1
                and tag_id != previous_tag_id
            )

        self.label_starts = np.frompyfunc(starts_entity, 2, 1)
        # x marks an entity end (last token) iff x is a B or I label such that the next token maps either to O, or starts a new entity
        self.label_ends = np.frompyfunc(
            lambda x, x_plus1: x > 0 and (
                x_plus1 == 0 or starts_entity(x_plus1, x)),
            2,
            1,
        )
        self.output = {}

    def setup(self, trainer, pl_module, stage):
        self.itotag = pl_module.model.config.id2label

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        labels, start_offsets, end_offsets = self.get_entities(
            prediction, batch["offsets"]
        )
        invnr = pl_module.tokenizer.decode(
            batch["invnr"], skip_special_tokens=True
        ).strip()
        if invnr not in self.output:
            self.output[invnr] = []
        doc = pl_module.tokenizer.decode(
            batch["doc"], skip_special_tokens=True).strip()
        self.output[invnr].append((doc, labels, start_offsets, end_offsets))

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._write_predictions(
            trainer.predict_dataloaders.dataset.typesystem,
        )

    def _write_predictions(self, typesystem):
        if not self.output:
            return
        make_entity = typesystem.get_type(NAMED_ENTITY)
        for invnr in self.output:
            with z.ZipFile(invnr) as izf:
                for doc, labels, begins, ends in self.output[invnr]:
                    output_path = os.path.join(self.outdir, doc)
                    output_doc_dir = os.path.join(
                        self.outdir, os.path.dirname(doc))
                    if not os.path.exists(output_doc_dir):
                        os.makedirs(output_doc_dir)
                    with izf.open(doc) as fi:
                        cas = load_cas_from_xmi(fi, typesystem=typesystem)
                        for label, begin, end in zip(labels, begins, ends):
                            cas.add(
                                make_entity(
                                    begin=begin.item(),
                                    end=end.item(),
                                    value=self.itotag[label.item()][2:],
                                )
                            )
                        cas.to_xmi(output_path)
        self.output.clear()

    def get_entities(self, token_labels, token_offsets):
        label_ids = np.append(
            torch.Tensor.cpu(token_labels),
            0,
        )  # add 0 to label_ids for shifting
        entity_starts = self.label_starts(
            label_ids, np.roll(label_ids, 1)
        ).nonzero()  # shift right to get previous tokens
        entity_ends = self.label_ends(
            label_ids, np.roll(label_ids, -1)
        ).nonzero()  # shift left to get next tokens
        entity_start_offsets = token_offsets[:, 0][entity_starts]
        entity_end_offsets = token_offsets[:, 1][entity_ends]
        entity_labels = label_ids[entity_starts]
        return entity_labels, entity_start_offsets, entity_end_offsets


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.tagset",
                              "model.init_args.tagset")
        parser.link_arguments(
            "data.init_args.pretrained_model", "model.init_args.pretrained_model"
        )


if __name__ == "__main__":
    cli = MyLightningCLI(save_config_kwargs={"overwrite": "true"})
