# Distillation
Distillation on classification problem implemented in Tensorflow

# Introduction
In this work I've implemented distillation network model based on [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531). 

The chosen task was a classification of 5 classes with flowers photo dataset. Firstly, `teacher` network was implemented with 2 million parameters,
then the `student` network with 1 million parameters.

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
* During the experiments all networks were trained during 15 epochs with batch size 32

# Experiments

1. The first step was to understand the performance of `teacher` and `student` networks fully trained on dataset. The performance of each network was measured by accuracy of its prediction on test dataset. 
Below you can see the results of these two networks **fully trained** on dataset:

Network | Test Accuracy
------------ | -------------
Teacher | 0.743
Student | 0.749

As we can see, `student` network performs a little bit better than a `teacher` both of them **fully trained** on dataset. `Student` network has two times less parameters than `teacher` but shows accuracy which is slightly better than `teacher`. 
It could be due to following factors:
* `teacher` network is too big for this classification task, so smaller `student` network has enough capacity to perform as good as `teacher`
* these networks are tested not on the same test set so it's possible to have little differences in accuracy. To fight this problem we can train and estimate our models by using cross validation technics 
* it's possible that the `teacher` network not the most optimal one for this kind of dataset so it's important to test different architectures

2. In the second part of experience different **temperature** parameter was tested:

Temperature | Test Accuracy
------------ | -------------
3 | 0.741
6 | 0.722
10 | 0.735
15 | 0.727
20 | 0.745

According to the obtained results we don't see a huge gain in performance by using different temperatures values. This can be connected with the fact that the proportion for two losses is the same (**alpha** = 0.5) and we need to move the importance closer to hard loss to boost the performance of the `student` network.

3. In the third part of experiment we've tried to reduce the student model as far as we can. So we've divided by 2 the number of neurons in the pre-last dense layer, the temperature parameter is the same **T**=3:

Number of neurons | Test Accuracy
------------ | -------------
Student - 64 (1M param)| 0.741
Student - 32 (600K param)| 0.734
Student - 16 (345K param)| 0.730

As we can see the accuracy goes slowly down when we reduce the number of parameters of the student network. It could be related to two factors. First - distillation helps us to teach `student` network very efficiently so even when we drastically reduce the number of parameters we still obtain good accuracy results. Second - the `teacher` network is not the most optimal network for that task (too complex) so smaller models perform even better. To know which variant is correct we need to make more experiments with `student` networks fully trained on dataset with the same architecture as above.

4. The last part of experiment was inspired by the [On the Effacy of KD](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf). The main concept behind this article is that the early stop in the training process of `teacher` could be a more representable and better example for the training process of `student` network. So in that experiment we've trained `teacher` model for 3 epochs (instead of 15) and then proceed to distillation.

Network | Test Accuracy
------------ | -------------
Teacher (3 epochs) | 0.643
Student (15 epochs)| 0.737

The results of this approach look prominent. As it was shown in the article that was discussed above, we don't need train `teacher` network for the whole cycle we just need to have few epochs and then we can launch directly the distillation process. In that experiment we obtain almost the same results as from fully trained `teacher` network. Few epochs training process learns `teacher` network to obtain a more complete and more erroneus distribution which has much more useful information for the `student` network. It could be interesting to to dive into and test this approach more.

# Conclusion

To conclude, in that work we've tested different distillation approaches: variation of temperature parameter, early-stop strategy and also tested different `student` distilated architectures. In my experiments I haven't succeded to show the importance of the **temperature** parameter, I think the problem might be connected with the choice of **alpha** parameter or with the choice of `teacher` architecture. Also it can be useful to test a bigger range of **temperature** parameter.

For the model compression test (third part of experiment) we can't tell precisely if achieved results are related to distillation success or just more correct models were applied. We need to make more experiments on that.

And the last part of experiment might shown us that we are not obliged to train the teacher for the whole training process we could do it just for several epochs and then directly go to the distillation process. 

In my opinion, in order to understand if the distillation process works in our task first of all we need to test many different `teacher` models in order to understand what works best and after it go to the distillation process. 

# Future work

For the future work it could be very interesting to test the following approaches:
* At first, it must be interesting to change the **alpha** parameter to test the importance of each loss. It would be very interesting to test some border cases where **alpha** equals 0 or 1
* During the `student` training process recuperate not only output of the `teacher` but also recuperate some middle values in order to learn the representation of `teacher` network more thoroughly 
* One of the very interesting experiments could be to try to estimate the possible network sizes of `teacher` and `student` which provide the best performance. From one side `teacher` should be complex enough to be able to give sufficient information to the `student` however as it was shown in [On the Effacy of KD](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf) very deep `teacher` networks tend to be not very good teachers
* Also it could be interesting to train `student` models with different configurations fully on the dataset in order to compare them with its distilated variants 
