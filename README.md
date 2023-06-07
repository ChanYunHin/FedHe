# FedHe: Heterogeneous Models and Communication-Efficient Federated Learning

This repository is the official implementation of [FedHe: Heterogeneous Models and Communication-Efficient Federated Learning](https://arxiv.org/abs/2110.09910). 

## Requirements

Tensorflow version 2.0

## Training and Evaluation

To train the model(s) of cifar10, run this command:

```train
sh run_cifar10.sh
```

To train the model(s) of mnist, run this command:

```train
sh run_mnist.sh
```

## Results

Our model achieves the following performance on :

#### Heterogeneous Models for CIFAR10 and MNIST under 10 clients

| Model name         | CIFAR-10  | MNIST |
| ------------------ |---------------- | -------------- |
| FedHe   |     62%         |      98.5%       |
| FedMD   |   57.5%         |       98%        |
| Private |    57%          |       98%        |


## Citation

If you find the paper provides some insights or our code useful, please consider giving a star ⭐ and citing:

```
@inproceedings{chan2021fedhe,
  title={Fedhe: Heterogeneous models and communication-efficient federated learning},
  author={Chan, Yun Hin and Ngai, Edith CH},
  booktitle={2021 17th International Conference on Mobility, Sensing and Networking (MSN)},
  pages={207--214},
  year={2021},
  organization={IEEE}
}
```
