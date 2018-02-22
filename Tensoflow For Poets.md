# 
run TensorFlow on a single machine, and will train a simple classifier to classify images of flowers.

We will be using transfer learning, which means we are starting with a model that has been already trained on another problem. We will then be retraining it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

We are going to use a model trained on the ImageNet Large Visual Recognition Challenge dataset. These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. You will have a choice of model architectures, so you can determine the right tradeoff between speed, size and accuracy for your problem.

### What you'll Learn
- [x] How to use Python and TensorFlow to train an image classifier
- [x] How to classify images with your trained classifier

### step1: Preparations

* Install TensorFlow
 
 Before we can begin the tutorial you need to install tensorflow.

> If you already have TensorFlow installed, be sure it is a recent version. This codelab requires at least version 1.2. You can upgrade to the most recent stable branch with

> pip install --upgrade tensorflow

* Clone the git repository

Clone the repository and cd into it. This is where we will be working.

```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
```
### step2: dataset(using your own dataset or downloading provided sample dataset)

Before you start any training, you'll need a set of images to teach the model about the new classes you want to recognize. We've created an archive of creative-commons licensed flower photos to use initially. Download the photos (218 MB) by invoking the following two commands:
```
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C tf_files
```
You should now have a copy of the flower photos in your working directory. Confirm the contents of your working directory by issuing the following command:
```
ls tf_files/flower_photos
```
The preceding command should display the following objects:
```
daisy/
dandelion/
roses/
sunflowers/
tulip/
LICENSE.txt
```
### step3: Configure your network

The retrain script can retrain either Inception V3 model or a MobileNet. In this exercise, we will use a MobileNet. The principal difference is that Inception V3 is optimized for accuracy, while the MobileNets are optimized to be small and efficient, at the cost of some accuracy.

Inception V3 has a first-choice accuracy of 78% on ImageNet, but is the model is 85MB, and requires many times more processing than even the largest MobileNet configuration, which achieves 70.5% accuracy, with just a 19MB download.

Pick the following configuration options:

> Input image resolution: 128,160,192, or 224px. Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy. We recommend 224 as an initial setting.
The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25. We recommend 0.5 as an initial setting. The smaller models run significantly faster, at a cost of accuracy.

With the recommended settings, it typically takes only a couple of minutes to retrain on a laptop. You will pass the settings inside Linux shell variables. Set those shell variables as follows:
