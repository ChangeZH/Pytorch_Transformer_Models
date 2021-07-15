import torch
import argparse
from tools import *
from utils import *
from models import *
from datasets import *


def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--project', type=str, default='ResT-Lite')
    parser.add_argument('--model_config', type=str, default='./configs/models/ResT/ResT-Lite.yaml')
    parser.add_argument('--dataset_config', type=str, default='./configs/datasets/CIFAR10.yaml')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_img_path', type=str, default='./datasets/CIFAR10/test/')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./work_dirs/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weight', type=str, default='./epoch_1.pth')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--milestones', type=list, default=[500, 750])

    args = parser.parse_args()
    return args


def run(args):
    model_configs = load_config(args.model_config)
    dataset_configs = load_config(args.dataset_config)
    train_datasets = Classification_Datasets(set_name='train', **dataset_configs)
    val_datasets = Classification_Datasets(set_name='val', **dataset_configs)
    model = eval(model_configs['model_type'])(**model_configs)

    if args.weight is not None:
        cp = torch.load(args.weight)
        model.load_state_dict(cp['model'].state_dict())
    else:
        pass

    if args.train:
        train(args=args, model=model, train_datasets=train_datasets, val_datasets=val_datasets)
    elif args.val:
        val(args=args, model=model, train_datasets=train_datasets, val_datasets=val_datasets)
    elif args.test:
        test(args=args, model=model, model_configs=model_configs, dataset_configs=dataset_configs)
    else:
        assert 'Please Select at Least One Mode in [--train, --val, --test]', (args.train or args.val or args.test)


if __name__ == '__main__':
    args = get_args()
    run(args=args)
