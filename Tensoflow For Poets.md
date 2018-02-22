# 
run TensorFlow on a single machine, and will train a simple classifier to classify images of flowers.

We will be using transfer learning, which means we are starting with a model that has been already trained on another problem. We will then be retraining it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

We are going to use a model trained on the ImageNet Large Visual Recognition Challenge dataset. These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. You will have a choice of model architectures, so you can determine the right tradeoff between speed, size and accuracy for your problem.

### What you'll Learn
- [x] How to use Python and TensorFlow to train an image classifier
- [x] How to classify images with your trained classifier

### step1: Preparations

* 1. Install TensorFlow
 
 Before we can begin the tutorial you need to install tensorflow.

> If you already have TensorFlow installed, be sure it is a recent version. This codelab requires at least version 1.2. You can upgrade to the most recent stable branch with

> pip install --upgrade tensorflow
