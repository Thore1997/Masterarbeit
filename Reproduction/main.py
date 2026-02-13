import torch
from data_loader import Data_Loader
from train import trainer
import argparse


def main():
    dl=Data_Loader()
    train_dataset, test_dataset, labels = dl.get_dataset(args.dataset[0])
    trainer_object=trainer(args)
    f_score,auc_score=trainer_object.train_and_evaluate(train_dataset,test_dataset,labels)
    return (f_score,auc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3000, help='batch size for training')
    #parser.add_argument('--dataset', type=str, default='wine', help='name of dataset')
    parser.add_argument('--dataset', type=str, nargs='+', default=['thyroid'], help='names of datasets')
    parser.add_argument('--faster_version', type=str, default='no', help='faster version with a lower number of repeats')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f1,auc = main()
    print("F1, AUC: ",f1, auc)


