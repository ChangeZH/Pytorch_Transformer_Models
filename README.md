# Pytorch_Transformer_Models
&emsp;&emsp;基于Pytorch框架的CV-Transformer，包含针对多种网络的复现。  
&emsp;&emsp;详细请访问 👉 https://blog.csdn.net/qq_36449741/article/details/118308150

## 环境要求 / Environmental Requirements  
```
conda create -n PTM python=3.7 -y  
conda activate PTM  
conda install pytorch torchvision cudatoolkit -c pytorch  
pip install tqdm pyyaml tensorboardX prettytable pillow einops  
```

## 支持模型 / Models
- [x]  ⚡ Swin Transformer 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118439062?spm=1001.2014.3001.5501)
- [x]  ⚡ NexT 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118308866?spm=1001.2014.3001.5501)
- [x]  ⚡ RexT 👉   链接：[CSDN_ChangeZH](https://blog.csdn.net/qq_36449741/article/details/118309006?spm=1001.2014.3001.5501)

## 参数设置 / Parameter Setting  

```python
--project # 项目名，即在--save_path下生成文件夹名称
--model_config # 模型配置文件
--dataset_config # 数据集配置文件
--train # 训练模式
--val # 验证模式
--test # 测试模式
--test_img_path # 测试文件夹路径
--topk # 测试结果保存为top-k，当k=5即top5
--devices # 运行gpu位置
--batch_size # 运行batch大小
--max_epochs # 训练最大代数
--val_interval # 每次测试、保存权重所隔代数
--save_path # 运行保存路径
--resume # 继续训练加载权重路径
--weight # 加载权重路径
--lr # 学习率
--gamma # 学习率衰减率
--milestones # 学习率衰减里程碑
```

## 运行 / Running

运行  ` python run.py --train `  进行训练。  
运行  ` python run.py --val `  进行验证。  
运行  ` python run.py --test `  进行测试。  
