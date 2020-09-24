# Distillation
Distillation on classification problem implemented in Tensorflow

# Introduction
In this work I've implemented distillation network model based on [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531). 

The chosen task was a classification of 5 classes with flowers photo dataset. Firstly, teacher network was implemented with 4 million parameters,
then the student network with 2 million parameters.

Once these two networks were created the next step was to combine them in a distillation pipeline. In distillation process we use already trained 
teacher model. In more details you can see its architecture below.

![Distillation pipeline](https://github.com/Denisoidd/Distillation/blob/master/images/architecture.PNG)

The main idea behind this pipeline is the following. We give an input (image in our case) to both networks 
`student` and `teacher`. `Teacher` is already trained and we **won't update its weights**. For `Student` 
model we **update its weights**. 

During the train process we will use two losses: **distillation**, **student**. They are multiplied by `alpha` and `(1 - alpha)` respectively where **alpha** = 0.5. The first one will help us to imitate the distribution of teacher model with **temperature** > 1. The bigger the temperature parameter is the "softer" distribution we will obtain after softmax activation. "Softer" distribution has more usefull information for student training process. 

The second one is a standart CrossEntropy loss for 5 classes, the same that was used for `teacher` network.

# How to run the code

The easiest and recommended way is to launch `experiments.ipynb` in Google Colab and just follow the instructions inside it.

If you want to train `teacher`, `student` or `student distillation` networks you should run: `teacher_train.py`, `student_train.py`, `distillation_train.py` respectively.

# Network architecture

As it was already mentioned we have three network architectures: `student`, `teacher` and `student distillation`. First two of them `student` and `teacher` are almost identical the main difference is on the pre-last dense layer. For the `student` we have two time less parameters than for the `teacher`. 

The network itself consists of the following layers:
* Data augmentation - horizontal flip, rotation, zoom
* Image rescaling
* 4 blocks of 2D convolution with max pooling. Number of filters [16, 32, 64, 128]
* Flatten and Dropout 
* Dense layer with 64 (for student) and 128 (for teacher) nodes
* Dense layer with 5 nodes (number of classes)

## Other network details

* As an optimizer Adam was used. 
* Data was divided into two parts (80% - train, 20% - test). Of course it would be better to divide dataset on three parts: test, val and train but it's not big enough (around 3500 images) so it was decided to have only test part.

# Experiments

The first step was to understand the performance of `teacher` and `student` networks fully trained on dataset. The performance of each network was measured by accuracy of its prediction on test dataset. 
Below you can see the results of these two networks **fully trained** on dataset:

In the second part of experience different **temperature** parameter was tested:

**WRITE:** Conclusion about temperature parameter in distillation problem

In the third part of experiment we've tried to reduce the student model as far as we can. So we've divided by 2 the number of nodes in the pre-last dense layer:

**WRITE:** Conclusion about size reduction parameter in distillation problem

The last part of experiment was inspired by the [On the Effacy of KD](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf). The main concept behind this article is that the early stop in the training process of `teacher` could be a more representable and better example for the training process of `student` network.

**WRITE:** Conclusion about early stop parameter in distillation problem

