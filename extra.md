# Part 1
The paper tries to use Cycle GAN to translate an image from a source domain X to a target domain Y in the absence of paired examples. Due to the unpaired nature of the training set, the learnt translation does not guarantee that an individual input x and output y are paired up in a meaningful way as there are infinitely many mappings G that will induce the same distribution over $\hat{y}$. Additionally, in practice, it is difficult to optimize the adversarial objective in isolation: standard procedures often lead to the well-known problem of mode collapse, where all input images map to the same output image and the optimization fails to make progress. Due to thse problems, cycle consistent loss is introduced to encourage $F(G(x)) \approx x$ and $G(F(y)) \approx y$.

Specifically, for a large-enough network, it can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. In this case, the cycle-consistent assumption is made to further reduce the space of possible mapping functions. On the other hand, the standard GAN loss is still necessary since it is the key to GANs’ success - the idea of an adversarial loss that forces the generated images to be, in principle, indistinguishable from real images. It is different from the newly introduced cycle loss as it aims at matching the distribution of generated images to the data distribution in the target domain while cycle consistent loss focuses on preventing the learned mappings G and F from contradicting each other.


# Part 2

Training GANs consists in finding a Nash equilibrium to a two-player non-cooperative game. Each player (discriminator and generator) wishes to minimize its own cost function. A Nash equilibirum is a point (θ(D), θ(G)) such that J(D) is at a minimum with respect to θ(D) and J(G) is at a minimum with respect to θ(G). Unfortunately, finding Nash equilibria is a very difficult problem, and thus GAN is notorious for difficult to train. 

The idea that a Nash equilibrium occurs when each player has minimal cost seems to intuitively motivate the idea of using traditional gradient-based minimization techniques to minimize each player’s cost simultaneously. Unfortunately, a modification to θ(D) that reduces J(D) can increase J(G), and a modification to θ(G) that reduces J(G) can increase J(D). Gradient descent thus fails to converge for many games. Many existent approaches to GAN training have thus applied gradient descent on each player’s cost simultaneously, despite the lack of guarantee that this procedure will converge.

Due to the high computational cost and lack of GPU resources, the model is only trained for 11 epochs, but it is sufficient to see effect. As three plots shown below, the GAN loss, Cycle loss and total loss for two mappings (G and F) are shown below. 

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/plots/gan_loss.png) | 
|:--:| 
| Training GAN Loss|

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/plots/cycle_loss.png) | 
|:--:| 
| Training Cycle Loss|

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/plots/total_loss.png) | 
|:--:| 
| Training Total Loss|

Additionally, ten images from test set are shown below for demonstration of the performance of the model.

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/img/10pairs.png) | 
|:--:| 
| 10 images from test set|

As expected, the change of loss is not stable and it seems that the two models have not started to converge at all. 

Based on emperical study, there are a number of popular tricks for training the GAN:

1. Normalize the inputs

- normalize the images between -1 and 1
- use Tanh as the last layer of the generator output

2. BatchNorm

- construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images
- when batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).

3. Avoid Sparse Gradients: ReLU, MaxPool

- LeakyReLU is a good choise
- for Downsampling, use: Average Pooling, Conv2d + stride
- for Upsampling, use: PixelShuffle, ConvTranspose2d + stride

4. Use the ADAM Optimizer

- use SGD for discriminator and ADAM for generator

5. Add noise to inputs, decay over time

- add some artificial noise to inputs to D
- add gaussian noise to every layer of generator

Due to lack of computational cost, it impossible for me to test all these ideas and tricks. I only experimented with changing the optimizer of discriminator from Adam to SGD. However, it is hardly to see any obvious improvement on the loss curve within limited epochs. It is likely that it needs more time to see effect.

# Part3

I would like to use GANs to detect the clothes of the person in a picture, and change texture and color of the clothes. In the process of clothes changes, it is expected that the hair and the skin of the person is not significantly changed.

## Data

The dataset is the In Shop Clothes Retrieval Benchmark by Deep Fashion from CUHK. The whole dataset contains 50000 images. For the purpose of testing the idea fast, only 1000 thousand images are selected and carefully divided into two groups - A: persons with long pants, B: persons with short pants.

## Architecture
### Original Cycle-GAN
Original Cycle-GAN from the paper is directly used. After training x epochs, the result is shown in the figure below. The result is far from satisfying. This is expected as the different between two training groups (i.e. long pants and short pants) are less obvious and easy to capture than those successful applications such as horse2zebra. 

### Add segmentation
In order to focus on the leg, rather than the whole image, a segmentation is used to highlight the leg of the person in the image. Specifically, an encoding-decoding based segmentation network [1] is used here to give 0 values to the leg and. A segmentation example is shown below.


By doing this, it will directly give hints to the discriminator. The Generative Loss Function is updated as followed:

$$ L_{GAN} = E_{y\sim p(y)}[log(D_Y(Seg(Y) * Y))] + E_{x\sim p(x)}[log(1 - D_Y(Seg(X) * F(X)))] $$

<!-- ![formula](http://latex2png.com/output//latex_21674e9e1cbb0d9e5d9b14268e7b0996.png) -->

### Add skin loss

A simple skin based loss is added so as to tell the discriminator which way to go when transferring between long pants and short pants. As the formula below shows, we transferring from long pants (denoted as X) to short pants (denoted as Y), the loss function below could give hints to the discriminator to converge towards a directed goal:

$$ L_{skin_X} = |skin_X - logits_X| $$

<!-- ![f1](http://latex2png.com/output//latex_be455ea4b05e53aa6c5fb88a6d000499.png) -->

$$ skin_X = \frac{S_X}{1+S_X}, S_X = relu(G(X) - origRGB)$$

$$logits_X = relu(-relu(skin_X))$$

## Experiments and results

Due to the limitation of time and resources, the model is only trained for few epochs. The figure below shows the example results with relatively satisfying appearance from test data. The model tends to have ability of preserving some attributes that are not expected to change and focus on the other attributes that are expected to change.

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/img/result.png) | 
|:--:| 
| Training GAN Loss for long2short|

| ![All models: Loss vs. Batch](https://raw.githubusercontent.com/PAN001/Cycle-GAN-Shorts-2-Leggings/master/plots/l2s_GAN_loss.png) | 
|:--:| 
| Training GAN Loss for long2short|

# Reference

[1] https://github.com/vanhuyz/CycleGAN-TensorFlow

https://github.com/soumith/ganhacks

https://github.com/shekkizh/neuralnetworks.thought-experiments

https://github.com/soumith/ganhacks

https://github.com/lemondan/HumanParsing-Dataset

