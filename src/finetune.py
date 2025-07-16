from lightning.pytorch.cli import LightningCLI
import os
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.tagset_path", "model.init_args.tagset_path"
        )
        parser.link_arguments("data.init_args.batch_size",
                              "model.init_args.batch_size")
        parser.link_arguments(
            "data.init_args.pretrained_model", "model.init_args.pretrained_model"
        )


def cli_main():
    cli = MyLightningCLI(save_config_kwargs={"overwrite": "true"})


if __name__ == "__main__":
    cli_main()
