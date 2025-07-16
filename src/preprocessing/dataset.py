import itertools
import json
import logging
import os
import random
from enum import Enum
from zipfile import ZipFile

import click
from cassis import load_cas_from_xmi, load_typesystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
NAMED_ENTITY = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"


class NerDataset:
    def __init__(
        self,
        datasetname,
        inputdata,
        outputdir,
        xmltypesystem,
        maxtokens,
    ):
        self.zip_file = ZipFile(inputdata)
        self.outputdir = outputdir
        self.datasetname = datasetname
        self.members = sorted(
            x.filename
            for x in self.zip_file.infolist()
            if not x.is_dir() and x.file_size > 0
        )
        with open(xmltypesystem, "rb") as f:
            self.typesystem = load_typesystem(f)
        self.max_nb_tokens = maxtokens
        self.data = {}
        self.report = {}

    def defined_split(self):
        for split in ["train", "validation", "test"]:
            docs = [m for m in self.members if split in m]
            if docs:
                self._assign(self._get_instances(docs), split)

    def random_split(self, trainratio, seed):
        instances = self._get_instances(self.members)
        random.seed(seed)
        random.shuffle(instances)
        trainlength = int(len(instances) * trainratio)
        self._assign(instances[:trainlength], "train")
        self._assign(instances[trainlength:], "validation")
        self.report = self._report()
        self.save()

    def _report(self):
        return {
            partition: summarize(self.data[partition]) for partition in self.data.keys()
        }

    def write_report(self, name):
        with open(os.path.join(self.outputdir, name), "w") as f:
            json.dump(self.report, f, ensure_ascii=False)
            f.write("")

    def _assign(self, instances, partition, exist_ok=False):
        if not exist_ok and partition in self.data:
            raise ValueError(
                "Data partition already exists, set `exist_ok` to True to overwrite."
            )
        tokens, labels = zip(*list(instances))
        self.data[partition] = {"tokens": list(tokens), "labels": list(labels)}

    def _extend(self, new_data, partition):
        if partition not in self.data:
            self.data[partition] = new_data
        else:
            self.data[partition]["tokens"].extend(new_data["tokens"])
            self.data[partition]["labels"].extend(new_data["labels"])

    def _get_instances(self, members):
        token_seqs, label_seqs = [], []
        for m in members:
            tokens, labels = extract_instances(
                self.zip_file, m, self.typesystem, self.max_nb_tokens
            )
            token_seqs.extend(tokens)
            label_seqs.extend(labels)
        return list(zip(token_seqs, label_seqs))

    def fold_split(self, kfolds, seed):
        instances = self._get_instances(self.members)
        random.seed(seed)
        random.shuffle(instances)
        fold_length = len(instances) // kfolds
        self.report = {}
        for i in range(kfolds):
            endval = min((i + 1) * fold_length, len(instances))
            val_data = instances[i * fold_length : endval]
            train_data = instances[0 : i * fold_length] + instances[endval:-1]
            self._assign(train_data, "train")
            self._assign(val_data, "validation")
            self.save(datasetsfx=str(i + 1))
            self.report[f"fold_{i + 1}"] = self._report()
            self.data = {}

    def doc_split(self, docsplit):
        with open(docsplit) as f:
            data_split = json.load(f)
        for partition in data_split.keys():
            instances = self._get_instances(data_split[partition])
            self._assign(instances, partition)
        self.save()
        self.report = self._report()

    def size(self, partition="train"):
        return len(self.data[partition]["tokens"])

    def labelset(self, partition):
        if partition in self.data:
            return set(x for seq in self.data[partition]["labels"] for x in seq)
        else:
            return set()

    def unknown_devtest_labels(self):
        train_labels = self.labelset("train")
        dev_labels = self.labelset("validation")
        test_labels = self.labelset("test")
        s = ""
        if dev_labels - train_labels:
            s += f"Validation data contains unknown labels: {dev_labels - train_labels}\n"
        if test_labels - train_labels:
            s += f"Test data contains unknown labels: {test_labels - train_labels}\n"
        if not s:
            return "Data contains no unknown labels"
        else:
            raise ValueError(s)

    def save(self, datasetsfx=""):
        """Write dataset to file"""
        os.makedirs(self.outputdir, exist_ok=True)
        with open(
            os.path.join(self.outputdir, f"{self.datasetname}{datasetsfx}.json"), "w"
        ) as f:
            json.dump(self.data, f)


def summarize(data):
    nb_tokens = sum(len(seq) for seq in data["tokens"])
    nb_entities = sum(
        len([x for x in seq if x.startswith("B-")]) for seq in data["labels"]
    )
    return {
        "sequences": len(data["tokens"]),
        "tokens": nb_tokens,
        "entities": nb_entities,
    }


def extract_instances(zip_file, m, typesystem, max_nb_tokens):
    token_seqs = []
    label_seqs = []

    with zip_file.open(m, "r") as f:
        logger.info(f"Processing {m}")
        cas = load_cas_from_xmi(f, typesystem=typesystem)
        for sentence in cas.select(SENTENCE):
            chunked_tokens, chunked_labels = extract_tokens_and_labels(
                sentence, cas, max_nb_tokens
            )
            token_seqs.extend(chunked_tokens)
            label_seqs.extend(chunked_labels)

    return token_seqs, label_seqs


def extract_tokens_and_labels(sentence, cas, max_nb_tokens):
    """Get tokens and labels covered by `sentence`.

    Sequences are chunked if their length exceeds `max_nb_tokens`"""
    tokens, labels = get_tokens_and_labels(sentence, cas)
    if not tokens:  # in case of empty lines
        return [], []
    chunked_tokens, chunked_labels = split_sequences(
        tokens, labels, max_nb_tokens, dest_tokens=[], dest_labels=[]
    )
    return chunked_tokens, chunked_labels


def split_sequences(tokens, labels, max_nb_tokens, dest_tokens=[], dest_labels=[]):
    while len(tokens) > max_nb_tokens:
        dest_tokens.append(tokens[:max_nb_tokens])
        tokens = tokens[max_nb_tokens:]
        dest_labels.append(labels[:max_nb_tokens])
        labels = labels[max_nb_tokens:]
    dest_tokens.append(tokens)
    dest_labels.append(labels)
    return dest_tokens, dest_labels


def split_seq_of_seqs(tokenseqs, labelseqs, max_nb_tokens):
    dest_tokens, dest_labels = [], []
    for tokens, labels in zip(tokenseqs, labelseqs):
        split_sequences(tokens, labels, max_nb_tokens, dest_tokens, dest_labels)
    return dest_tokens, dest_labels


class BIO(Enum):
    O = "O"
    B = "B"
    I = "I"

    @classmethod
    def begin(cls, label):
        if label is None:
            return "O"
        return f"B-{label}"

    @classmethod
    def midword(cls, label):
        if label is None:
            return "O"
        return f"I-{label}"


def get_tokens_and_labels(sentence, cas):
    """get tokens and entities covered by the sentence, mapping entities to BIO labels"""
    tokens = cas.select_covered(TOKEN, sentence)
    entities = cas.select_covered(NAMED_ENTITY, sentence)

    labels = []
    i = 0
    for e in entities:
        while i < len(tokens) and tokens[i].begin < e.begin:
            labels.append(BIO.O.value)
            i += 1
        labels.append(BIO.begin(e.value))
        i += 1
        while i < len(tokens) and tokens[i].end <= e.end:
            labels.append(BIO.midword(e.value))
            i += 1
    labels.extend(
        list(
            itertools.chain.from_iterable(
                itertools.repeat(BIO.O.value, len(tokens) - i)
            )
        )
    )
    return [x.get_covered_text() for x in tokens], labels


def write_report(data, report_path):
    report = {partition: summarize(data[partition]) for partition in data.keys()}
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False)
        f.write("")


@click.group()
@click.option("-i", "--inputdata", type=click.Path(exists=True))
@click.option("-o", "--outputdir", type=click.Path(), default="data/A")
@click.option(
    "-x",
    "--xmltypesystem",
    type=click.Path(exists=True),
    default="resources/TypeSystem.xml",
)
@click.option("-m", "--maxtokens", default=240)
@click.pass_context
def cli(ctx, inputdata, outputdir, xmltypesystem, maxtokens):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj["inputdata"] = inputdata
    ctx.obj["outputdir"] = outputdir
    ctx.obj["xmltypesystem"] = xmltypesystem
    ctx.obj["maxtokens"] = maxtokens


@cli.command()
@click.pass_obj
@click.option("-d", "--datasetname", default="traindata")
@click.option("-s", "--seed", default=42)
@click.option("-r", "--trainratio", default=0.8)
@click.option("-n", "--reportname", default="report_random.json")
def rndm(ctx, datasetname, seed, trainratio, reportname):
    ds = NerDataset(datasetname, **ctx)
    ds.random_split(trainratio, seed)
    ds.write_report(reportname)


@cli.command()
@click.pass_obj
@click.option("-d", "--datasetname", default="traindata_")
@click.option("-s", "--seed", default=42)
@click.option("-k", "--kfolds", default=5)
@click.option("-r", "--reportname", default="report_fold.json")
def fold(ctx, datasetname, seed, kfolds, reportname):
    ds = NerDataset(datasetname, **ctx)
    ds.fold_split(kfolds, seed)
    ds.write_report(reportname)


@cli.command()
@click.pass_obj
@click.option("-d", "--datasetname", default="traindata_doc")
@click.option(
    "-s",
    "--splitdict",
    type=click.Path(exists=True),
    help="a json dictionary mapping split names to list of zip xmi members",
)
@click.option("-r", "--reportname", default="report_doc.json")
def docsplit(ctx, datasetname, splitdict, reportname):
    ds = NerDataset(datasetname, **ctx)
    ds.doc_split(splitdict)
    ds.write_report(reportname)


@cli.command()
@click.pass_obj
@click.option("-a", "--augmentation_dataset", default="data/A/traindata_3.json")
@click.option("-d", "--augmented_datasetname", default="AB")
@click.option("-s", "--base_outputdir", default="data/B")
@click.option("-b", "--base_datasetname", default="B")
@click.option("-r", "--reportnamepfx", default="data_report")
def augment_with_predefined_split(
    ctx,
    base_outputdir,
    base_datasetname,
    augmented_datasetname,
    augmentation_dataset,
    reportnamepfx,
):
    with open(augmentation_dataset) as f:
        aug_ds = json.load(f)
    ds = NerDataset(base_datasetname, **ctx)
    ds.outputdir = base_outputdir
    ds.defined_split()
    ds._extend(aug_ds["validation"], "validation")
    ds.report = ds._report()
    ds.save()
    ds.write_report(f"{reportnamepfx}_{base_datasetname}.json")

    ds = NerDataset(augmented_datasetname, **ctx)
    ds.defined_split()
    ds._extend(aug_ds["validation"], "validation")
    ds._extend(aug_ds["train"], "train")
    ds.report = ds._report()
    ds.save()
    ds.write_report(f"{reportnamepfx}_{augmented_datasetname}.json")


if __name__ == "__main__":
    cli(obj={})
