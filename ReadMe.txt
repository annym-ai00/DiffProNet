# **Diffusion Prototypical Network for Few-Shot Fine-Grained Classification**

## Environment
The code is tested on Windows 11 with Anaconda3 and following packages:
- python 3.11.5
- pytorch 2.4.0

Install the required dependencies:
```
pip install -r requirements.txt
```

## Preparation
1. Change the ROOT_PATH value in the following file to yours:
    - `datasets/datasets.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - **CUB-200-2011**: download from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/), rename the file to `CUB_200_2011` and put in `data` folder.

    - **Stanford Dogs**: download from [here](https://github.com/ayushdabra/stanford-dogs-dataset-classification), rename the file to `Stanford_Dogs` and put in `data` folder.

    - **Stanford Cars**: download from [here](https://github.com/cyizhuo/Stanford_Cars_dataset), rename the file to `Stanford_Cars` and put in `data` folder.



## Experiments:
To train on 5-way 1-shot and 5-way 5-shot Stanford Dogs:

```
python train.py --dataset dog --save-path ./model_trained --shot 1
python train.py --dataset dog --save-path ./model_trained --shot 5
```

To evaluate on 5-way 1-shot and 5-way 5-shot Stanford Dogs:
```
python test.py --dataset dog --shot 1 --save-path ./model_trained
python test.py --dataset dog --shot 5 --save-path ./model_trained
```
