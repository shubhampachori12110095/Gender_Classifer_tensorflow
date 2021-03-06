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

##### If you have the git repository from the first Tensorflow for Poets:

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

##### If you don't have the git repository from the first Tensorflow for Poets:
```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

```
### step 2: Test the model
Next, verify that the model is producing sane results before starting to modifying it.

The scripts/ directory contains a simple command line script, label_image.py, to test the network. Now we'll test label_image.py on this picture of some daisies:

Now test the model. If you are using a different architecture you will need to set the "--input_size" flag.
```
python -m scripts.label_image \
  --graph=tf_files/retrained_graph.pb  \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg
```
The script will print the probability the model has assigned to each flower type. Something like this:
```
Evaluation time (1-image): 0.140s

daisy 0.7361
dandelion 0.242222
tulips 0.0185161
roses 0.0031544
sunflowers 8.00981e-06
```
### step 3: Optimize the model
Mobile devices have significant limitations, so any pre-processing that can be done to reduce an app's footprint is worth considering.

**Limited libraries on mobile**

One way the TensorFlow library is kept small, for mobile, is by only supporting the subset of operations that are commonly used during inference. This is a reasonable approach, as training is rarely conducted on mobile platforms. Similarly it also excludes support for operations with large external dependencies. You can see the list of supported ops in the [tensorflow/contrib/makefile/tf_op_files.txt](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/tf_op_files.txt) file. 

By default, most graphs contain training ops that the mobile version of TensorFlow doesn't support. TensorFlow won't load a graph that contains an unsupported operation (even if the unsupported operation is irrelevant for inference).

**Optimize for inference**

To avoid problems caused by unsupported training ops, the TensorFlow installation includes a tool, optimize_for_inference, that removes all nodes that aren't needed for a given set of input and outputs.

The script also does a few other optimizations that help speed up the model, such as merging explicit batch normalization operations into the convolutional weights to reduce the number of calculations. This can give a 30% speed up, depending on the input model. Here's how you run the script:
```
python -m tensorflow.python.tools.optimize_for_inference \
  --input=tf_files/retrained_graph.pb \
  --output=tf_files/optimized_graph.pb \
  --input_names="input" \
  --output_names="final_result"
```
Running this script creates a new file at tf_files/optimized_graph.pb.

**Verify the optimized model**

To check that optimize_for_inference hasn't altered the output of the network, compare the label_image output for retrained_graph.pb with that of optimized_graph.pb:
```
python -m scripts.label_image \
  --graph=tf_files/retrained_graph.pb\
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg
```
```
python -m scripts.label_image \
    --graph=tf_files/optimized_graph.pb \
    --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg
```
When I run these commands I see no change in the output probabilities to 5 decimal places.

Now run it yourself to confirm that you see similar results.

**Investigate the changes with TensorBoard**

If you followed along for the first tutorial, you should have a tf_files/training_summaries/ directory (otherwise, just create the directory by issuing the following Linux command: mkdir tf_files/training_summaries/).

The following two commands will kill any runninng TensorBoard instances and launch a new instance, in the background watching that directory:
```
pkill -f tensorboard
tensorboard --logdir tf_files/training_summaries &
```
TensorBoard, running in the background, may occasionally print the following warning to your terminal, which you may safely ignore

WARNING:tensorflow:path ../external/data/plugin/text/runs not found, sending 404.

Now add your two graphs as TensorBoard logs:
```
python -m scripts.graph_pb2tb tf_files/training_summaries/retrained \
  tf_files/retrained_graph.pb 

python -m scripts.graph_pb2tb tf_files/training_summaries/optimized \
  tf_files/optimized_graph.pb 
```
Now open TensorBoard, and navigate to the "Graph" tab. Then from the pick-list labeled "Run"on the left side, select "Retrained". 

Explore the graph a little, then select "Optimized" from the "Run" menu.

From here you can confirm some nodes have been merged to simplify the graph. You can expand the various blocks by double-clicking them.

### step 4: Make the model compressible

**Check the compression baseline**

The retrained model is still 84MB in size at this point. That large download size may be a limiting factor for any app that includes it.

Every mobile app distribution system compresses the package before distribution. So test how much the graph can be compressed using the gzip command:
```
gzip -c tf_files/optimized_graph.pb > tf_files/optimized_graph.pb.gz

gzip -l tf_files/optimized_graph.pb.gz
```
```
            compressed        uncompressed    ratio     uncompressed_name
            5028302             5460013       7.9%      tf_files/optimized_graph.pb
```
Not much! :relieved:

On its own, compression is not a huge help. For me this only shaves 8% off the model size. If you're familiar with how neural networks and compression work this should be unsurprising.

The majority of the space taken up by the graph is by the weights, which are large blocks of floating point numbers. Each weight has a slightly different floating point value, with very little regularity.

But compression works by exploiting regularity in the data, which explains the failure here.

**Example: Quantize an Image**

Images can also be thought of as large blocks of numbers. One simple technique for compressing images it to reduce the number of colors. You will do the same thing to your network weights, after I demonstrate the effect on an image.

Below I've used [ImageMagick's](https://www.imagemagick.org/script/index.php) convert utility to reduce an image to 32 colors. This reduces the image size by more than a factor of 5 (png has built in compression), but has degraded the image quality.
<div align="center">
<img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/68b7f947c4e09b3e.png" height="300px" alt="24 bit color: 290KB" title="24 bit color: 290KB"> <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/7551be7b2cd5e1bb.png" height="300px" alt="32 colors: 55KB" title="32 colors: 55KB" >
</div>

**Quantize the network weights**

Applying an almost identical process to your neural network weights has a similar effect. It gives a lot more repetition for the compression algorithm to take advantage of, while reducing the precision by a small amount (typically less than a 1% drop in precision).

It does this without any changes to the structure of the network, it simply quantizes the constants in place.

Now use the quantize_graph script to apply these changes:

(This script is from the TensorFlow repository, but it is not included in the default installation)

```
python -m scripts.quantize_graph \
  --input=tf_files/optimized_graph.pb \
  --output=tf_files/rounded_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```
Now try compressing this quantized model:

```
gzip -c tf_files/rounded_graph.pb > tf_files/rounded_graph.pb.gz
gzip -l tf_files/rounded_graph.pb.gz
```
```
         compressed        uncompressed   ratio    uncompressed_name
            1633131             5460032   70.1%    tf_files/rounded_graph.pb
```
You should see a significant improvement. I get 70% compression instead of the 8% that gzip provided for the original model.

Now before you continue, verify that the quantization process hasn't had too negative an effect on the model's performance.

First manually compare the two models on an example image.
```
python -m scripts.label_image \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg \
  --graph=tf_files/optimized_graph.pb
python -m scripts.label_image \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg \
  --graph=tf_files/rounded_graph.pb
```
For me, on this input image, the output probabilities have each changed by less than one tenth of a percent (absolute).

Next verify the change on a larger slice if the data to see how it affects overall performance.

First evaluate the performance of the baseline model on the validation set. The last two lines of the output show the average performance. It may take a minute or two to get the results back.
```
python -m scripts.evaluate  tf_files/optimized_graph.pb
```
For me, optimized_graph.pb scores scores 90.9% accuracy, and 0.270 for cross entropy error (lower is better).

Now compare that with the performance of the model in rounded_graph.pb:
```
python -m scripts.evaluate  tf_files/rounded_graph.pb
```
You should see less than a 1% change in the model accuracy.

These differences are far from statistically significant. The goal is simply to confirm that the model was clearly not broken by this change.

### step 5: Setup the Android app

**Install AndroidStudio**

If you don't have it installed already, go install [AndroidStudio 3.0+](https://developer.android.com/studio/index.html).

**Open the project with AndroidStudio**
Open a project with AndroidStudio by taking the following steps:

1. Open AndroidStudio. After it loads select "Open an existing Android Studio project" from this popup:
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/1482ddc7911df61b.png">
</center>

2. In the file selector, choose tensorflow-for-poets-2/android/tfmobile from your working directory.
3. You will get a "Gradle Sync" popup, the first time you open the project, asking about using gradle wrapper. Click "OK".
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/b9f9a03dd27fd1bb.png" height="300px">
</center>

### step 6: Test run the app

The app can run on either the a real Android device, or in the AndroidStudio Emulator.

**Set up an Android device**

You can't load the app from android studio onto your phone unless you activate "developer mode" and "USB Debugging". This is a one time setup process.

Follow these [instructions](https://developer.android.com/studio/debug/dev-options.html#enable).

**Or set up the emulator with camera access**

Android studio makes setting up an emulator easy. Since this app uses the camera, you may want want to setup the emulator's camera to use your computer's camera instead of the default test pattern. 

To do this you need to create a new device in the "Android Virtual Device Manager", which you can access with this button. From the main ADVM page select "Create VIrtual Device":
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/4a7a52c6d04a350b.png" height="300px">
</center>

Then on the "Verify Configuration" page, the last page of the virtual device setup, select "Show Advanced Settings":
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/fc6fc5d70c77f5f2.png" height="300px">
</center>

With the advanced settings shown, you can set the camera source to use the host computer webcam:
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/14b9c0f2666819bc.png" height="300px">
</center>

**Test Build and install the app**
Before making any changes to the app let's run the version that ships with the repository.
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/774326d4e89c2559.png">
</center>

Run a Gradle sync, and then hit play, in Android Studio to start the build and install process.

Next you will need to select your phone from this popup:
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/b315baa6b4a3e80f.png" height="200px">
</center>

Now allow the Tensorflow Demo to access your camera and files:
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/7b8fced048328e6c.png" height="200px">
</center>

Now that the app is installed, click the app icon, to launch it. This version of the app uses the standard MobileNet, pre-trained on the 1000 ImageNet categories. It should look something like this ("Android" is not one of the available categories)::
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/9b607aec2f4567c8.png" height="300px">
</center>

### step 7: Run the customized app

The default app setup classifies images into one of the 1000 ImageNet classes, using the standard MobileNet, without the retraining we did in part 1.

Now let's modify the app so that the app will use our retrained morel for our custom image categories.

**Add your model files to the project**

The demo project is configured to search for a graph.pb, and a labels.txt files in the android/tfmobile/assets directory. Replace those two files with your versions. The following command accomplishes this task:
```
cp tf_files/rounded_graph.pb android/tfmobile/assets/graph.pb
cp tf_files/retrained_labels.txt android/tfmobile/assets/labels.txt 
```
**Change the output_name in ClassifierActivity.java**

The TensorFlow Interface used by the app requires that you ask for your results by name. The app is currently set up to read the output of the baseline MobileNet, named "MobilenetV1/Predictions/Softmax". The output node for our model has a different name: "final_result". Open ClassifierActivity.java and update the OUTPUT_NAME variable as follows:
```
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "final_result";
```
**Run your app**

In Android Studio run a Gradle sync, , so the build system can find your files, and then hit play, , in to start the build and install process as before.

It should look something like this:
<center>
    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/49ef495e892ee86f.png" height="600px">
</center>

You can hold the power and volume-down buttons together to take a screenshot.

Now try a web search for flowers, point the camera at the computer screen, and see if those pictures are correctly classified.

> If you get a Gradle sync error:
> <center>
>    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/186d73f2db74031.png" height="300px">
> </center>
>It's because gradle couldn't find android/tfmobile/assets/graph.pb, or android/tfmobile/assets/labels.txt. Verify the locations of those files and rerun the gradle sync by clicking the "Sync Project with Gradle Files" button from the toolbar:
> <center>
>    <img src="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/img/774326d4e89c2559.png">
> </center>


