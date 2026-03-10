import torch
from data_loader import Data_Loader
from train import trainer
import argparse
import argparse
import torch

def main(args):
    dl = Data_Loader()
    train_dataset, test_dataset, labels = dl.get_dataset(args.dataset[0])
    trainer_object = trainer(args)
    f1_score, auc_score, auprc_score = trainer_object.train_and_evaluate(train_dataset, test_dataset, labels)
    return (f1_score, auc_score, auprc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3000, help='batch size for training')
    parser.add_argument('--dataset', type=str, nargs='+', default=['wineori'], help='names of datasets')
    parser.add_argument('--faster_version', type=str, default='no', help='faster version with a lower number of repeats')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f1_score, auc_score, auprc_score = main(args)

    print(f"--- Final Results ---")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"ROC-AUC:   {auc_score:.4f}")
    print(f"AUPRC:     {auprc_score:.4f}")