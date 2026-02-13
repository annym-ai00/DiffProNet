Diffusion-ProtoNet: 
Diffusion Prototypical Network for Few-Shot Fine-Grained Classification

# Dataset Structure:
dataset/
|- CUB_200_2011/
|	|- train/
|	|- valid/
|	|- test/
|- Stanford_Dogs/
|	|- train/
|	|- valid/
|	|- test/
|- Stanford_Cars/
|	|- train/
|	|- valid/
|	|- test/


# How to Run:
1. Train Model:
> python train.py --dataset dog --save-path ./model_trained

2. Test Model:
> python test.py --dataset dog --save-path ./model_trained


