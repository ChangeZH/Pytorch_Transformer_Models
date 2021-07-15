import torch
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader


def val(args, model, val_datasets):
    val_dataloader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
    num_iter_per_epoch = len(val_dataloader)

    obj_list = val_datasets.id_labels

    model.eval()
    val_acc = 0
    val_acc_dict = {obj_list[i]: {'correct': 0, 'wrong': 0, 'other_wrong': 0} for i in obj_list}
    with tqdm(total=num_iter_per_epoch) as val_bar:
        for iter, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, results = torch.max(outputs.softmax(-1), dim=1)
            val_acc += (results == labels).sum()

            for j in range(labels.shape[0]):
                label = obj_list[labels[j].item()]
                result = obj_list[results[j].item()]
                if result == label:
                    val_acc_dict[label]['correct'] += 1
                else:
                    val_acc_dict[label]['wrong'] += 1
                    val_acc_dict[result]['other_wrong'] += 1

            val_bar.set_description(
                '\t VAL. Iter: {}/{}. Acc: {:.4f} '.format(iter + 1, num_iter_per_epoch, val_acc / num_iter_per_epoch))
            val_bar.update(1)
    show_accuracy(val_acc_dict)
    return val_acc
