import click
import itertools as it
import json
import logging
import os
import sys
import urllib.request

from dataset import split_seq_of_seqs, write_report

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

TRAIN_URL = "https://data.yoda.vu.nl:9443/vault-fgw-llc-vocmissives/voc_gm_ner%5B1670857835%5D/original/datasplit_all_standard/train.conll"
DEV_URL = "https://data.yoda.vu.nl:9443/vault-fgw-llc-vocmissives/voc_gm_ner%5B1670857835%5D/original/datasplit_all_standard/dev.conll"
TEST_URL = "https://data.yoda.vu.nl:9443/vault-fgw-llc-vocmissives/voc_gm_ner%5B1670857835%5D/original/datasplit_all_standard/test.conll"
LABEL_MAPPING = {
    "LOC": "LOC_NAME",
    "LOCderiv": "LOC_ADJ",
    "PER": "PER_NAME",
    "RELderiv": "ETH_REL",
    "SHP": "SHIP",
}


def missing_classes(base_tagsetpath, vocgm_tagsetpath):
    with open(base_tagsetpath) as f:
        base_tagset = json.load(f)
    with open(vocgm_tagsetpath) as f:
        vocgm_tagset = json.load(f)

    return [
        v
        for k, v in base_tagset.items()
        if not any(y == k for y in map_labels(vocgm_tagset.keys(), LABEL_MAPPING))
    ]


def missing_file(url, partition, cachedir):
    return url is not None and not os.path.exists(
        os.path.join(cachedir, f"{partition}.conll")
    )


def download(cachedir, train_url, dev_url, test_url):
    for url, split in [
        (train_url, "train"),
        (dev_url, "validation"),
        (test_url, "test"),
    ]:
        if missing_file(url, split, cachedir):
            logging.info(f"Downloading {url}")
            download_file(url, os.path.join(cachedir, f"{split}.conll"))


def download_file(url, outpath):
    with urllib.request.urlopen(url) as f:
        data = f.read().decode("utf-8")
        with open(outpath, "w") as of:
            of.write(data)


def flush(tokens, token_seqs, labels, label_seqs):
    token_seqs.append([t for t in tokens])
    label_seqs.append([x for x in labels])
    tokens.clear()
    labels.clear()


def conll2json(conll, maxtokens=-1):
    logging.info(f"Converting {conll} to jsonl")
    with open(conll) as f:
        token_seqs, label_seqs = [], []
        tokens, labels = [], []
        for line in f.readlines():
            elts = line.strip().split()
            if len(elts) > 1:
                tokens.append(elts[0])
                labels.append(elts[1])
            elif tokens:
                flush(tokens, token_seqs, labels, label_seqs)
        if tokens:
            flush(tokens, token_seqs, labels, label_seqs)
    if maxtokens > 0:
        token_seqs, label_seqs = split_seq_of_seqs(token_seqs, label_seqs, maxtokens)

    return {"tokens": token_seqs, "labels": label_seqs}


def map_labels(inputlabels, label_mapping):
    labels = []
    for x in inputlabels:
        if x != "O" and x[2:] in label_mapping:
            labels.append(x[0:2] + label_mapping[x[2:]])
        else:
            labels.append(x)
    return labels


def convert(outputdir, output, reportname, maxtokens):
    d = {"train": [], "validation": [], "test": []}
    for split in ["train", "validation", "test"]:
        d[split] = conll2json(os.path.join(outputdir, f"{split}.conll"), maxtokens)

    with open(os.path.join(outputdir, output), "w") as f:
        json.dump(d, f, ensure_ascii=False)

    write_report(d, os.path.join(outputdir, reportname))


def add_unk_classes(data, unk_classes=[]):
    data["train"]["unk_classes"] = list(
        it.repeat(unk_classes, len(data["train"]["tokens"]))
    )
    data["validation"]["unk_classes"] = list(
        it.repeat(unk_classes, len(data["validation"]["tokens"]))
    )


def entity_density(labelseq):
    return sum(x != "O" for x in labelseq) / len(labelseq)


def add_sequences(augment_data, data, threshold, unk_classes):
    augment_tokens = augment_data["tokens"]
    augment_labels = augment_data["labels"]
    if threshold > 0:
        augment_tokens, augment_labels = list(
            zip(
                *[
                    (ts, ls)
                    for (ts, ls) in zip(augment_tokens, augment_labels)
                    if entity_density(ls) > threshold
                ]
            )
        )

    data["tokens"].extend(augment_tokens)
    data["labels"].extend([map_labels(seq, LABEL_MAPPING) for seq in augment_labels])
    data["unk_classes"].extend(list(it.repeat(unk_classes, len(augment_tokens))))
    return data["tokens"], data["labels"]


@click.group()
def cli():
    pass


@cli.command()
@click.option("-o", "--outputdir", type=click.Path(), default="data/vocgm")
@click.option("-d", "--datasetname", type=click.Path(), default="vocgm.json")
@click.option("-r", "--reportname", type=click.Path(), default="vocgm_report.json")
@click.option("-m", "--maxtokens", default=240)
def create(outputdir, datasetname, reportname, maxtokens):
    os.makedirs(outputdir, exist_ok=True)
    download(outputdir, TRAIN_URL, DEV_URL, TEST_URL)
    convert(outputdir, datasetname, reportname, maxtokens)


@cli.command()
@click.option(
    "-i",
    "--input_data_path",
    default="data/A/traindata_3.json",
    help="path to reference dataset",
)
@click.option(
    "-a",
    "--augmentation_data_path",
    default="data/vocgm/vocgm.json",
    help="path to augmentation dataset (vocgm)",
)
@click.option(
    "-o",
    "--outdir",
    default="data/A_vocgm",
    help="output directory",
)
@click.option("-d", "--datasetname", type=click.Path(), default="A_vocgm.json")
@click.option(
    "-r",
    "--reportname",
    default="A_vocgm_report.json",
    help="name of dataset report",
)
@click.option(
    "-s",
    "--ner_tagsetpath",
    type=click.Path(),
    default="resources/tagsets/ner_tagset.json",
)
@click.option(
    "-t",
    "--vocgm_tagsetpath",
    type=click.Path(),
    default="resources/tagsets/vocgm_tagset.json",
)
@click.option(
    "-e",
    "--entity_density_threshold",
    default=0,
    type=float,
    help="filters augmentation sequences by their entity density if threshold exceeds zero",
)
def augment(
    input_data_path,
    augmentation_data_path,
    outdir,
    datasetname,
    reportname,
    ner_tagsetpath,
    vocgm_tagsetpath,
    entity_density_threshold,
):
    with open(input_data_path) as f:
        data = json.load(f)
    with open(augmentation_data_path) as f:
        augment_data = json.load(f)
    add_unk_classes(data)
    tokenseqs, labelseqs = add_sequences(
        augment_data["train"],
        data["train"],
        float(entity_density_threshold),
        unk_classes=missing_classes(ner_tagsetpath, vocgm_tagsetpath),
    )
    data["train"]["tokens"] = tokenseqs
    data["train"]["labels"] = labelseqs
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, datasetname), "w") as f:
        json.dump(data, f, ensure_ascii=False)

    write_report(data, os.path.join(outdir, reportname))


if __name__ == "__main__":
    cli()
