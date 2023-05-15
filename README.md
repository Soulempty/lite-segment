## 数据准备
```
custom_dataset
    |
    |--images           # 存放所有原图
    |  |--train
    |  |  |--1.jpg
    |  |  |--...
    |  |--test
    |  |--val
    |
    |--labels           # 存放所有标注图
    |  |--train
    |  |  |--1.png
    |  |  |--...
    |  |--test
    |  |--val
```

## 模型训练
```shell
python train.py
```
## 模型测试与评估
```shell
python test.py
```

## 失败案例分析
```shell
python checkbad.py
```