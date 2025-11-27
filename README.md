# Fine-grained Named Entity Recognition

This repository contains the data and code for the paper *Fine-grained Named Entity Recognition for the 
East-India Company Domain*, which is currently under publication.

## Data

### Annotations
Annotations are provided in [./data/annotations/](./data/annotations/) as two zip files, reflecting the paper's data splits 
and experimental setup.


### Guidelines
Guidelines for the annotations can be found under [./docs/ner-guidelines.md](./docs/ner-guidelines.md)

### Training data
Training datasets for finetuning can be generated from the annotations with [DVC](https://dvc.org/).
Activate / install a local environment:

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements
```

Then run:

```bash
dvc repro
```

This will reproduce the steps specified in [dvc.yaml](./dvc.yaml), and generate training
data in [data/](./data/).

### Additional resources

The globalise training data corresponding to the `train A+B` set is available on HuggingFace as [globalise/globalise_NER_token_classification_dataset](https://huggingface.co/datasets/globalise/globalise_NER_token_classification_dataset).

## Code

The finetuning code uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). See [./cfg/](./cfg) for examples config files and [./scripts/](./scripts/) for applications.

### Additional resources

The test checkpoints of two models are available on HuggingFace:

* [globalise/globalise_NER_token_classification](https://huggingface.co/globalise/globalise_NER_token_classification): model finetuned with gloBERTise on the `train A+B` sets 
* [globalise/globalise_vocgm_NER_token_classification](https://huggingface.co/globalise/globalise_vocgm_NER_token_classification): model finetuned with gloBERTise on the globalise data (`train A` and `train B`) augmented with the training data of the [voc-gm-ner](https://research.vu.nl/en/datasets/voc-gm-ner-corpus/) corpus


## Citation


```
@article{10.63744@DRbhWNTzqNzR,
  title = {Fine-grained Named-Entity Recognition for the East-India Company domain},
  author = {Sophie Arnoult and Brecht Nijman and Leon van Wissen},
  year = {2025},
  journal = {Anthology of Computers and the Humanities},
  volume = {3},
  pages = {953--967},
  editor = {Taylor Arnold, Margherita Fantoli, and Ruben Ros},
  doi = {10.63744/DRbhWNTzqNzR}
}
```


