# Pytorch_Transformer_Models
&emsp;&emsp;基于Pytorch框架的CV-Transformer，包含针对多种网络的复现。  
&emsp;&emsp;详细请访问 👉 https://blog.csdn.net/qq_36449741/article/details/118308150

## 环境要求 / Environmental Requirements  
```
conda create -n PTM python=3.7 -y  
conda activate PTM  
conda install pytorch torchvision cudatoolkit -c pytorch  
pip install tqdm pyyaml tensorboardX opencv-python  
```

## 支持模型 / Models
- [x]  ⚡ Swin Transformer 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118439062?spm=1001.2014.3001.5501)
- [x]  ⚡ NexT 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118308866?spm=1001.2014.3001.5501)
- [x]  ⚡ RexT 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118309006?spm=1001.2014.3001.5501)

## 参数设置 / Parameter Setting  

```python
--project #项目名，即在
--save_path #下生成文件夹名称
--model_config
--dataset_config
--train
--val
--test
--test_img_path
--topk
--devices
--batch_size
--max_epochs
--val_interval
--save_path
--resume
--weight
--lr
--gamma
--milestones
```

## 训练 / Training  

运行  ` python run.py --train `  进行训练。  
