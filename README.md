# PerceptualLossExperiments
### Analyze the impact of perceptual losses on different computer vision tasks and methods.

## Introcudction
Perceptual losses like VGG-loss were reported to be much more correlated to human perception of image similariry than standard losses like L2.
Replacing L2 loss in tasks like autoencoders or super resolution models lead to better looking results.

This work continues the work of Dan Amir [Understanding and Simplifying Perceptual Distances](https://openaccess.thecvf.com/content/CVPR2021/papers/Amir_Understanding_and_Simplifying_Perceptual_Distances_CVPR_2021_paper.pdf)
Where he analyzes the VGG-loss as a kernel-MMD between patch distribution in the input images and proposes a new non-parametric loss 'MMD++''
which is comparable to VGG perceptualy while being much simpler, faster and extandable.

In this repository I perform varius experiment that mostly aim at comparing VGG perceptual losses to patch distribution losses trying to verify Dan's results
and extend them to more practical high-end tasks like image generation.

## 1. Adobe 2 Alternative Forced choice test (2AFC)
The first clue about the nature of VGG-loss comes from the Adobe-2AFC test:
originaly proposed at [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/pdf/1801.03924.pdf) this test is referenced in Dan's work as well. The Adobe-2AFC dataset contains patch triplets labeld by humans for which one of the first two is closer to the third perceptualy. These annotation are ued as GT for perceptual loses to be tested on.
As opposed to the original paper were randomly initialized VGG works worse than trained VGG Dan shows a simple variant random VGG that acheives comparable reults.
The MMD++ loss as well.

*2AFC sample*             |  *results table*
:------------------------:|:-----------------------:
![](assets/2afc-ref.png)  |  ![](assets/2afc.png)


## 2. Perceptual mean optimization
This type of experiment also appear in Dan's work under the name Generalized Image Mean (GIM). The mean of a set of images in L2 is a blurry unrealistic one. Optimizing for the mean of the same images while using VGG-loss as a metric leads to much smooth and perceptually good looking results. This leads to a set of experiments tha allow comparing the results of optimization  a mea image with different image losses.
Here, in some cases MMD++ shows a better performance than VGG.

*2 sets of 6 similar images*                                   | *results with different losses*  
:-----------------------------------------------------------:  |:-----------------------------------------------------------: 
<img src="assets/GIM__inputs.png" alt="drawing" width="500"/>  | <img src="assets/GIM__results.png" alt="drawing" width="500"/> 


## 3. Perceptual clustering.
cluster datasets using kmeans/one-hot-autoencoders while using perceptual distance metrics

## 4. Style transfer.
The most promiment line of work in neural style transfer surged by a serie of paper by Gatis et Al [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
The key concept is using L2 distane between VGG feature maps/ Gram matrices of those feature maps in order to conserve content/style accordingly.
This is very much related to the concept of perceptual loss but experiments with random VGG network show poor results.

*content + style| *VGG pretrained*    | *VGG random*  
:---------------:|:-----------:|:-----------:|
<img src="assets/Style_transfer5.png" alt="drawing" width="500"/>|<img src="assets/Style_transfer3.png" alt="drawing" width="250"/>|<img src="assets/Style_transfer4.png" alt="drawing" width="500"/> 


#### Normalized Gram matrix of style image
 *VGG pretrained*    | *VGG random*  
:---------------:|:-----------:|
<img src="assets/Style_transfer1.png" alt="drawing" width="500"/> | <img src="assets/Style_transfer2.png" alt="drawing" width="500"/>

## 5. Generative models.
Train autoencoder/[GLO](https://arxiv.org/abs/1707.05776) with VGG-loss instead of L2 is known to work better. Here I show this and try to acheive comparable results with random VGGs and MMD++.
Below are train-set reconstruction results of encoders (a DCGan generator and a similar encoder) trained on 128x128 FFHQ dataset with different losses


*L2*                                                         | *VGG pretrained*  
:-----------------------------------------------------------:  |:-----------------------------------------------------------: 
<img src="assets/AE_L2.png" alt="drawing" width="300"/>  | <img src="assets/AE_VGG-Pt.png" alt="drawing" width="300"/> 

*VGG random          *                                     | *MMD++*  
:-----------------------------------------------------------:  |:-----------------------------------------------------------: 
<img src="assets/AE_VGG-rand.png" alt="drawing" width="300"/>  | <img src="assets/AE_MMD++(P=3).png" alt="drawing" width="300"/> 
<!-- 
| Loss                  |Reconstruction                                                 |
|-----------------------|---------------------------------------------------------------|
| L2                    |<img src="assets/AE_L2.png" alt="drawing" width="300"/>  |
| VGG-random            |<img src="assets/AE_VGG-rand.png" alt="drawing" width="300"/>  |
| VGG-pretrained        |<img src="assets/AE_VGG-Pt.png" alt="drawing" width="300"/>  |
| MMD++                 |<img src="assets/AE_MMD++(P=3).png" alt="drawing" width="300"/>  | -->
