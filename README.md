# Model inversion attack on neural network

Implementation of the model inversion attack on the Gated-Recurrent-Unit neural network (GRU) using Generative adversarial network (GAN) for samples generation. May be used to determine the membership of training images or to determine the complete set of training data.

GAN is used for creation proxy images for feeding the target system as input. Further, images classified with a high degree of confidence by the target system can be assigned to a member of the training sample used to train this system. A neural network of several GRU layers targetted to classification problem is used as a target system, for which, a mechanism for protecting random data noise is implemented in addition to the standard functionality. 

The cifar10 dataset was used to train the target system from which only images of cats and dogs were pre-filtered. The same dataset with filtered dog images was used to train the generative gun network used to generate the images used to carry out the attack.

Results. Generated images confidently classified by the target system (left) and their possible prototypes from the training set of the target system (right):

![generated_plot_319_26](https://user-images.githubusercontent.com/39829227/209410044-588bf2a6-d95e-4bbd-aab6-cc7299221103.png)
![dog2](https://user-images.githubusercontent.com/39829227/209410105-1f29ddc2-9d98-483e-b1d9-672e0ceb4852.png)

![generated_plot_319_41](https://user-images.githubusercontent.com/39829227/209410058-5ae42b15-6f94-4b68-8186-8f42bba16ea0.png)
![dog7](https://user-images.githubusercontent.com/39829227/209410111-a1106572-bcb2-4656-8e5e-c5da3bd73d14.png)



