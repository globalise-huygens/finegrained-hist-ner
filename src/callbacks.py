import json
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
import os
from seqeval.metrics import classification_report
import torch


class RawPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("epoch")
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


class BestPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("batch")
        self.output_dir = output_dir
        self.best_token_predictions = []

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
        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        words_per_sequence = torch.count_nonzero(batch["token_mask"], dim=1)
        sequences = torch.split(prediction[token_starts], words_per_sequence.tolist())
        self.best_token_predictions.extend(torch.argmax(x, dim=1) for x in sequences)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        i2tag = pl_module.tagset.keys()
        with open(os.path.join(self.output_dir, "predictions.txt"), "w") as f:
            for seq in self.best_token_predictions:
                f.write(" ".join(i2tag[i] for i in seq) + "\n")


class TeacherPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("batch")
        self.output_dir = output_dir

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
        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        torch.save(
            prediction[token_starts],
            os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"),
        )


def to_label_string(seqs, i2tag):
    return [[i2tag[i] for i in seq] for seq in seqs]


class SeqevalWriter(BasePredictionWriter):
    def __init__(self, output_dir, name="seqeval_report.txt", write_predictions=False):
        super().__init__("batch")
        self.output_dir = output_dir
        self.name = name
        self.write_predictions = write_predictions
        self.best_token_predictions = []
        self.gold_token_labels = []

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
        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        words_per_sequence = torch.count_nonzero(batch["token_mask"], dim=1)
        prediction_sequences = torch.split(
            prediction[token_starts], words_per_sequence.tolist()
        )
        self.best_token_predictions.extend(
            torch.argmax(x, dim=1) for x in prediction_sequences
        )
        true_labels = torch.split(
            batch["labels"][token_starts], words_per_sequence.tolist()
        )
        self.gold_token_labels.extend(true_labels)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        i2tag = list(pl_module.tagset.keys())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, self.name), "w") as f:
            f.write(
                classification_report(
                    to_label_string(self.gold_token_labels, i2tag),
                    to_label_string(self.best_token_predictions, i2tag),
                    digits=3,
                )
            )
        if self.write_predictions:
            with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
                json.dump(
                    {
                        "true": to_label_string(self.gold_token_labels, i2tag),
                        "predictions": to_label_string(
                            self.best_token_predictions, i2tag
                        ),
                    },
                    f,
                )
