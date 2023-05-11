# UCL_model

Re-code some static architectural models in Continual Learning for evaluation and comparison


Source: [Uncertainty-based Continual Learning with Adaptive Regularization (UCL)](https://papers.nips.cc/paper/8690-uncertainty-based-continual-learning-with-adaptive-regularization) - published at NeurIPS 2019


#### To run UCL on split-CIFAR10/100, enter the following command:

```
$ python3 main.py --experiment split_cifar10_100 --approach ucl --conv-net --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3
```

