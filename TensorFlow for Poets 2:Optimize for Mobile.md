# TensorFlow for Poets 2:Optimize for Mobile

TensorFlow is a multipurpose machine learning framework. TensorFlow can be used anywhere from training huge models across clusters in the cloud, to running models locally on an embedded system like your phone.

### What you'll Learn
- [x] In TensorFlow for Poets: How to train a custom image recognition model
- [x] How to optimize your model.
- [x] How to compress your model.
- [x] How to run it in a pre-made Android app.

### What you will build
A simple camera app that runs a TensorFlow image recognition program to identify flowers.

### step 1: some preparations

If you have the git repository from the first Tensorflow for Poets:

We will be working in that same git directory, ensure that it is your current working directory, and check the contents, as follows:
```
cd tensorflow-for-poets-2
ls
```
This directory should contain three other subdirectories:

> * The android/tfmobile/ directory contains all the files necessary to build the a simple Android app that classifies images as it reads them from the camera. The only files missing for the app are those defining the image classification model, which you will create in this tutorial.
> * The scripts/ directory contains the python scripts you'll be using throughout the tutorial. These include scripts to prepare, test and evaluate the model.
> * The tf_files/ directory contains the files you should have generated in the first part. At minimum you should have the following files containing the retrained tensorflow program:
> ```
> retrained_graph.pb  retrained_labels.txt
> ```
