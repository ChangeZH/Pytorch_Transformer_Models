import os
import yaml
import prettytable as pt
from tensorboardX import SummaryWriter


def load_config(file_name):
    with open(file_name, 'r') as f:
        config = yaml.safe_load(f)
        return config


def init_tensorboard(args):
    if not os.path.exists(f'{args.save_path}/{args.project}'):
        os.mkdir(f'{args.save_path}/{args.project}')

    if args.train:
        command = 'train'
    elif args.val:
        command = 'val'
    elif args.test:
        command = 'test'
    else:
        exit()

    if not os.path.exists(f'{args.save_path}/{args.project}/{command}'):
        os.mkdir(f'{args.save_path}/{args.project}/{command}')
    if command != 'train':
        return None
    else:
        writer = SummaryWriter(logdir=f'{args.save_path}/{args.project}/{command}/tensorboard/', comment='run')
        print(f'Run Tensorboard:\n tensorboard --logdir={args.save_path}/{args.project}/{command}/tensorboard/')

        return writer


def show_accuracy(accuracy):
    Acc = {i: {'correct': accuracy[i]['correct'], 'wrong': accuracy[i]['wrong'],
               'other_wrong': accuracy[i]['other_wrong']} for i in accuracy}
    tb = pt.PrettyTable()
    tb.field_names = ['Class_Name', 'TP', 'FN', 'FP', 'P', 'R']
    correct = 0
    wrong = 0
    other_wrong = 0
    for i in accuracy:
        correct += accuracy[i]['correct']
        wrong += accuracy[i]['wrong']
        other_wrong += accuracy[i]['other_wrong']
        P = accuracy[i]['correct'] / (accuracy[i]['correct'] + accuracy[i]['wrong']) \
            if accuracy[i]['correct'] + accuracy[i]['wrong'] != 0 else 0
        R = accuracy[i]['correct'] / (accuracy[i]['correct'] + accuracy[i]['other_wrong']) \
            if accuracy[i]['correct'] + accuracy[i]['other_wrong'] != 0 else 0
        tb.add_row([i, accuracy[i]['correct'], accuracy[i]['wrong'], accuracy[i]['other_wrong'], P, R])
        Acc[i].update({'P': P, 'R': R})
    AP = correct / (correct + wrong) if correct + wrong != 0 else 0
    AR = correct / (correct + other_wrong) if correct + other_wrong != 0 else 0

    tb.add_row([' ', correct, wrong, other_wrong, AP, AR])
    print(tb)

    Acc.update({'ALL': {'correct': correct, 'wrong': wrong, 'other_wrong': other_wrong, 'P': AP, 'R': AR}})
    return Acc, tb


if __name__ == '__main__':
    config = load_config('CIFAR100.yaml')
    print(config)
