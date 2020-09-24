# Distillation
Distillation on classification problem implemented in Tensorflow

# Introduction
In this work I've implemented distillation network model based on [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531). 

The chosen task was a classification of 5 classes with flowers photo dataset. Firstly, teacher network was implemented with 4 million parameters,
then the student network with 2 million parameters.

Once these two networks were created the next step was to combine them in a distillation pipeline. In distillation process we use already trained 
teacher model. In more details you can see its architecture below.

