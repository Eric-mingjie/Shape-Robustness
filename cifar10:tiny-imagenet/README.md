# Adv Edge
## Prerequisite
### Required Python Packages



## CNNEdge
### CIFAR-10
```
CUDA_VISIBLE_DEVICES=[GPUs to USE] python cifar_cnnedge.py

```

### Tiny-ImageNet
```
CUDA_VISIBLE_DEVICES=[GPUs to USE] python tiny_cnnedge.py

```

## EdgeNetRob
### CIFAR-10
```
CUDA_VISIBLE_DEVICES=[GPUs to USE] python train_edgenet_cifar10.py \
     --epochs 160 --schedule 80 120 --train-batch 128 --thres 0.3 --sigma 1.0 \
     --high_threshold 0.3 --low_threshold 0.2 \
     --save [PATH to SAVE the RESULTS] \
     --cnnedge_path [PATH to LOAD the CNNEDGE]
```
### Tiny-ImageNet
```
CUDA_VISIBLE_DEVICES=[GPUs to USE] python train_edgenet_tiny.py \
     --epochs 90  --train-batch 128 --thres 0.3 --sigma 1.0 \
     --high_threshold 0.3 --low_threshold 0.2 \
     --save [PATH to SAVE the RESULTS] \
     --cnnedge_path [PATH to LOAD the CNNEDGE]
     --data_path [PATH to DATA]
```

### ImageNet
```
CUDA_VISIBLE_DEVICES=[GPUs to USE] python train_edgenet_imagenet.py \
     --epochs 90  --train-batch 128 --thres 0.3 --sigma 1.0 \
     --high_threshold 0.3 --low_threshold 0.2 \
     --save [PATH to SAVE the RESULTS] \
     --cnnedge_path [PATH to LOAD the CNNEDGE]
     --data_path [PATH to DATA]
```
where  
`--thres` is the parameter used in robust canny;  
`--sigma` is the parameter in original canny (basically the variance of Gaussian smoothing);  
`--high_threshold` and `--low_threshold` are also parameters in the original canny;
These parameters may not be the best for CIFAR10, so you may try different values for them;




## EdgeGanRob



### CIFAR-10
```
CUDA_VISIBLE_DEVICES=[GPUs to USE]  python3 train_edgegan_cifar.py --alpha 1 --beta 1.0 \
    --sigma 1.0 --high_threshold 0.2 --low_threshold 0.1 --thres 0.2 --update_D 1 \
    --save [PATH to SAVE the RESULTS]
    --gan_path [PATH to the TRAINED GAN]
```

### Tiny-ImageNet
```
CUDA_VISIBLE_DEVICES=[GPUs to USE]  python3 train_edgegan_tiny.py --alpha 1 --beta 1.0 \
    --sigma 1.0 --high_threshold 0.3 --low_threshold 0.2 --thres 0.3 --update_D 1 \
    --save [PATH to SAVE the RESULTS]
    --cnnedge_path [PATH to LOAD the CNNEDGE]
    --data_path [PATH to DATA]
    --gan_path [PATH to the TRAINED GAN]
```

where `alpha` is the parameter for l1 loss and `beta` is the parameter for classification loss.