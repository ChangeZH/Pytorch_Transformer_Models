import torch
from models import *

if __name__ == '__main__':
    model_list = {'SwinT': 224, 'ResT': 224, 'NesT': 32}
    for m in model_list:
        inputs = torch.ones(3, 3, model_list[m], model_list[m])
        model = eval(m)()
        outputs = model(inputs)
        print(m, 'input_size:', inputs.shape, 'output_size:', outputs.shape)
