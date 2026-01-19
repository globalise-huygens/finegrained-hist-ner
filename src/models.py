import json
import lightning as L
import torch
from torchmetrics.classification import MulticlassF1Score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForTokenClassification


def define_model(num_labels, freeze_model_params, pretrained_model, use_safetensors):
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        use_safetensors=use_safetensors,
    )
    model.train()
    if freeze_model_params:
        freeze_all_but_classifier(model)
    return model


def freeze_all_but_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


class NERModel(L.LightningModule):
    def __init__(
        self,
        tagset_path="resources/tagsets/globalise_tagset.json",
        learning_rate=1e-5,
        pretrained_model="globalise/gloBERTise",
        freeze_model_params=False,
        batch_size=32,
        reallocate_unseen_class_pmass=True,
        use_safetensors=True,
        monitor_scheduler=True,
    ):
        super().__init__()
        with open(tagset_path) as f:
            self.tagset = json.load(f)
        self.num_labels = len(self.tagset)
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, add_prefix_space=True
        )
        self.model = define_model(
            self.num_labels, freeze_model_params, pretrained_model, use_safetensors
        )
        self.batch_size = batch_size

        self.val_loss_fn = torch.nn.CrossEntropyLoss()

        if reallocate_unseen_class_pmass:
            self.loss_fn = torch.nn.NLLLoss()
            self.compute_loss_fn = self.compute_loss_with_unk_class_reallocation
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.compute_loss_fn = self.compute_loss

        self.micro_f1 = MulticlassF1Score(num_classes=self.num_labels, average="micro")
        self.micro_corr_f1 = MulticlassF1Score(
            num_classes=self.num_labels, average="micro"
        )
        self.macro_corr_f1 = MulticlassF1Score(
            num_classes=self.num_labels,
            average="macro",
        )
        self.weighted_corr_f1 = MulticlassF1Score(
            num_classes=self.num_labels,
            average="weighted",
        )
        self.monitored_scheduler = monitor_scheduler
        self.save_hyperparameters(ignore=["tagset_path"])

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def compute_loss(self, batch):
        outputs = self(**batch)
        return self.loss_fn(
            outputs["logits"].reshape(-1, self.num_labels),
            batch["labels"].view(-1),
        )

    def compute_loss_with_unk_class_reallocation(self, batch):
        outputs = self(**batch)
        log_softmax = torch.nn.LogSoftmax(dim=2)
        softmax = torch.nn.Softmax(dim=2)
        y_sm = softmax(outputs["logits"])

        # computes probability mass of unknown classes
        # batch and sequence dimensions are permuted for broadcast with batch-level class mask
        unk_pmass = torch.permute(
            torch.sum(
                torch.where(batch["unk_classes"], torch.permute(y_sm, (1, 0, 2)), 0),
                dim=2,
            ),
            (1, 0),
        )
        # adds unk-class probability mass to class 0 (index of 'O' NER label)
        add_y = torch.zeros(*y_sm.size(), device=self.device)
        add_y[:, :, 0] = torch.log(unk_pmass + y_sm[:, :, 0])
        # use updated log probability for class 0 and original log probabilities for other classes
        # computes log_softmax rather than log(softmax) for other classes
        y_lsm = log_softmax(outputs["logits"])
        y = torch.where(add_y < 0, add_y, y_lsm)

        loss = self.loss_fn(
            y.reshape(-1, self.num_labels),
            batch["labels"].view(-1),
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss_fn(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs["logits"]
        loss = self.val_loss_fn(
            logits.reshape(-1, self.num_labels),
            batch["labels"].view(-1),
        )
        self.log("val_loss", loss)

        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        predictions = torch.argmax(logits[token_starts], 1)
        true_labels = batch["labels"][token_starts]
        self.micro_f1(predictions, true_labels)
        self.log("micro_f1", self.micro_f1)

        predictions, true_labels = self.correct_zero_predictions(batch, logits)

        self.micro_corr_f1(predictions, true_labels)
        self.log("micro_corr_f1", self.micro_corr_f1)
        self.weighted_corr_f1(predictions, true_labels)
        self.log("weighted_corr_f1", self.weighted_corr_f1)
        self.macro_corr_f1(predictions, true_labels)
        self.log("macro_corr_f1", self.macro_corr_f1)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs["logits"]
        predictions, true_labels = self.correct_zero_predictions(batch, logits)
        self.micro_corr_f1(predictions, true_labels)
        self.log("micro_corr_f1", self.micro_corr_f1)
        self.weighted_corr_f1(predictions, true_labels)
        self.log("weighted_corr_f1", self.weighted_corr_f1)
        self.macro_corr_f1(predictions, true_labels)
        self.log("macro_corr_f1", self.macro_corr_f1)

    def correct_zero_predictions(self, batch, logits):
        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        predictions = torch.argmax(logits[token_starts], 1)
        true_labels = batch["labels"][token_starts]
        # filter out cases where both predictions and true label are 0
        m1 = torch.where(predictions > 0, True, False)
        m2 = torch.where(true_labels > 0, True, False)
        m = torch.logical_or(m1, m2)
        predictions = predictions[m]
        true_labels = true_labels[m]
        return predictions, true_labels

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        # outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs["logits"]
        token_starts = batch["token_mask"].nonzero(as_tuple=True)
        predictions = torch.argmax(
            logits[token_starts], 1
        )  # 1-dim tensor, with length of token starts in batch
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.monitored_scheduler:
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3
                ),
                "monitor": "micro_corr_f1",
            }
        else:
            lr_scheduler = {"scheduler": StepLR(optimizer, step_size=15, gamma=0.5)}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
