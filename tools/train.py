import os
import torch
from utils import *
from .val import val
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def train(args, model, train_datasets, val_datasets):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    writer = init_tensorboard(args=args)

    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    num_iter_per_epoch = len(train_dataloader)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.resume is not None:
        cp = torch.load(args.resume)
        start_epoch = cp['epoch']
        model.load_state_dict(cp['model'].state_dict())
        optimizer.load_state_dict(cp['optimizer'].state_dict())
        scheduler.load_state_dict(cp['scheduler'].state_dict())
    else:
        start_epoch = 1

    model = nn.DataParallel(model).cuda()
    obj_list = train_datasets.id_labels

    all_iter = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        print('Epoch: {}/{}'.format(epoch, args.max_epochs))
        model.train()
        train_acc = 0
        loss_epoch = 0
        train_acc_dict = {obj_list[i]: {'correct': 0, 'wrong': 0, 'other_wrong': 0} for i in obj_list}
        with tqdm(total=num_iter_per_epoch) as train_bar:
            for iter, (inputs, labels) in enumerate(train_dataloader):
                all_iter += 1
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, results = torch.max(outputs.softmax(-1), dim=1)
                train_acc += (results == labels).sum()

                for j in range(labels.shape[0]):
                    label = obj_list[labels[j].item()]
                    result = obj_list[results[j].item()]
                    if result == label:
                        train_acc_dict[label]['correct'] += 1
                    else:
                        train_acc_dict[label]['wrong'] += 1
                        train_acc_dict[result]['other_wrong'] += 1

                loss_batch = loss_func(outputs, labels)
                loss_epoch += loss_batch
                train_bar.set_description(
                    '\t TRAIN. Iter: {}/{}. Acc: {:.4f} Loss: {:.4f}. '.format(iter + 1, num_iter_per_epoch,
                                                                               train_acc / len(train_datasets),
                                                                               loss_epoch / num_iter_per_epoch))
                writer.add_scalars('Cls_Loss', {args.project + '_Train': loss_batch.item()}, global_step=all_iter)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                train_bar.update(1)
        writer.add_scalars('Accuracy', {args.project + '_Train': (train_acc / num_iter_per_epoch).item()},
                           global_step=epoch)
        show_accuracy(train_acc_dict)
        scheduler.step()
        if epoch % args.val_interval == 0:
            val_acc = val(args=args, model=model, val_datasets=val_datasets)
            torch.save({'model': model.module, 'epoch': epoch, 'optimizer': optimizer, 'scheduler': scheduler,
                        'id_labels': obj_list}, f'{args.save_path}/{args.project}/train/epoch_{epoch}.pth')
            writer.add_scalars('Accuracy', {args.project + '_Val': (val_acc / len(val_datasets)).item()},
                               global_step=epoch)
    return 0
