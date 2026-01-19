# Fine-grained Named Entity Recognition

This repository contains the data and code for NER model training and inference used for Globalise. 

## Data

### Annotations
#### Model validation
Annotations are provided in [./data/annotations/](./data/annotations/) as two zip files, reflecting the [paper](10.63744/DRbhWNTzqNzR)'s data splits 
and experimental setup.

#### Inference training
Annotations used for training the inference model, [globalise-NER](https://huggingface.co/globalise/globalise-NER), can be found in [./data/annotations/inference_training.zip](./data/annotations/inference_training.zip)

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
### Finetuning
The finetuning code uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). See [./cfg/](./cfg) for examples config files and [./scripts/](./scripts/) for applications.

### Inference
Inference over the OBP corpus is run with the [globalise/globalise-NER](https://huggingface.co/globalise/globalise-NER) model using [./src/inference.py](./src/inference.py) [./cfg/inference_invnr.yaml](./cfg/inference_invnr.yaml). Input data take the form of a directory of zipped XMI files, where XMI files of a same inventory number are zipped together. 

### Additional resources

The test checkpoints of two models are available on HuggingFace:

* [globalise/globalise_NER_token_classification](https://huggingface.co/globalise/globalise_NER_token_classification): model finetuned with gloBERTise on the `train A+B` sets 
* [globalise/globalise_vocgm_NER_token_classification](https://huggingface.co/globalise/globalise_vocgm_NER_token_classification): model finetuned with gloBERTise on the globalise data (`train A` and `train B`) augmented with the training data of the [voc-gm-ner](https://research.vu.nl/en/datasets/voc-gm-ner-corpus/) corpus

The inference model is available as [globalise/globalise-NER](https://huggingface.co/globalise/globalise-NER).

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


