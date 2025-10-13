# NER token classification

## Preprocessing

Annotations are provided in `./data/annotations/`.
Training datasets can be generated from this annotations with [DVC](https://dvc.org/).
Activate / install the local environment:

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements
```

Then run:

```bash
dvc repro
```

This will reproduce the steps specified in `dvc.yaml`, and generate training
data in `data/`.

## Finetuning

Finetuning uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). See `./cfg` for some config files and `./scripts/` for applications.

