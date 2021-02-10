from __future__ import print_function, division
from util.torch_util import BatchTensorToVars
from model.network import WeakMatchNet
from util.dataloader import DataLoader
from data.butterfly import Butterfly
import torch
from parser.parser import ArgumentParser


args, arg_groups = ArgumentParser(mode="eval").parse()

torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()

GPU = 0
NUM_WORKERS = 4
BATCH_SIZE = 4


def init_match_model(model_path, arg_groups):

    model = WeakMatchNet(use_cuda=use_cuda, **arg_groups["model"])

    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage
    )

    for name, param in model.FeatureExtraction.state_dict().items():
        try:
            model.FeatureExtraction.state_dict()[name].copy_(
                checkpoint["state_dict"]["FeatureExtraction." + name]
            )
        except KeyError:
            model.FeatureExtraction.state_dict()[name].copy_(
                checkpoint["FeatureExtraction." + name]
            )

    for name, param in model.FeatureRegression.state_dict().items():
        try:
            model.FeatureRegression.state_dict()[name].copy_(
                checkpoint["state_dict"]["FeatureRegression." + name]
            )
        except KeyError:
            model.FeatureRegression.state_dict()[name].copy_(
                checkpoint["FeatureRegression." + name]
            )

    for name, param in model.FeatureRegression2.state_dict().items():
        try:
            model.FeatureRegression2.state_dict()[name].copy_(
                checkpoint["state_dict"]["FeatureRegression2." + name]
            )
        except KeyError:
            model.FeatureRegression2.state_dict()[name].copy_(
                checkpoint["FeatureRegression2." + name]
            )

    return model


def init_eval_data():
    dataset = Butterfly(csv_file="data/csv/butterfly/eval.csv")

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return dataset, data_loader


def visualize(aff_AB, tps_Awrp_B):
    pass


def main():
    model_path = args.model
    model = init_match_model(model_path, arg_groups)
    model.eval()

    _, dataloader = init_eval_data()
    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    for batch in dataloader:
        batch = batch_tnf(batch)
        aff_AB, tps_Awrp_B = model(batch, training=False)
        visualize(aff_AB, tps_Awrp_B)


if __name__ == "__main__":
    main()
