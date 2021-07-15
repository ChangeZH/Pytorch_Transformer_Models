import os
import time
import math
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms


def test(args, model, model_configs, dataset_configs):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    if args.weight is not None:
        cp = torch.load(args.weight)
        obj_list = cp['id_labels']
        model.load_state_dict(cp['model'].state_dict())
    else:
        assert 'Test Mode need Model Weight File', args.weight is not None
    model = nn.DataParallel(model).cuda()

    test_img_list = [os.path.join(args.test_img_path, i) for i in os.listdir(args.test_img_path)]
    num_iter = math.floor(len(os.listdir(args.test_img_path)) / args.batch_size)
    transform = transforms.Compose([transforms.Resize((dataset_configs['input_size'], dataset_configs['input_size'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    model.eval()
    all_results_string = ''
    t = time.localtime()
    results_txt = open(f'results_{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.txt', 'w')
    with tqdm(total=num_iter + 1) as test_bar:
        for iter in range(num_iter + 1):
            test_bar.update(1)
            batch_img_list = test_img_list[iter * args.batch_size:(iter + 1) * args.batch_size]
            if len(batch_img_list) == 0:
                break
            imgs = [Image.open(img).convert('L') for img in batch_img_list] if dataset_configs[
                                                                                   'input_channel'] == 1 else [
                Image.open(img).convert('RGB') for img in batch_img_list]
            imgs = torch.stack([transform(img).cuda() for img in imgs], dim=0)
            outputs = model(imgs)
            confidences, results = torch.sort(outputs.softmax(-1), dim=1, descending=True)
            for j in range(imgs.shape[0]):
                batch_results_string = batch_img_list[j]
                for k in range(min(dataset_configs['num_classes'], args.topk)):
                    batch_results_string += '\t' + obj_list[results[j, k].item()] + '\t' + str(confidences[j, k].item())
                all_results_string += batch_results_string + '\n'
            test_bar.set_description('\t TEST. Iter: {}/{}. '.format(iter + 1, num_iter + 1))
    results_txt.write(all_results_string)
    results_txt.close()
    return 0
