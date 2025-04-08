# Furry-Art-Detector
A Pytorch project for binary classification of furry and non furry images

# Whats in each folder
### already_trained:
contains a python file for batch sorting an entire folder thats dragged and dropped on it (or provided in command-line)
the model included has been trained for 130 epochs, with 1,629 furry images, and 2,808 non-furry images on my 8GB 3050

furry images include various artstyles and forms from comics to sketches, although it contains some realistic artstyles, it picks up on cartoonish styles and sketches the most due to the dataset

non-furry images include various random screenshots from games, comic strips, random images from [this dataset](https://www.kaggle.com/datasets/ezzzio/random-images), and some CCTV footage from [this other dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)


its fairly robust, and tuning it to *not* overfit was a challenge.
you can fool the included model partially by just drawing lots of triangles, but it will be uncertain so setting a higher cutoff usually avoids false classification.
if you include some fake out data like that in your own training, it will most likely result in a much more robust model.

the trained images were all SFW, and the model hasnt been tested on any non-SFW images and results will vary if thats what you are using.

### for_data_processing:
contains a python file for converting and augmenting entire folders of images into the correct resolution for training (256,256) as well as making flipped copies, and copies with light noise
just drag and drop a folder on top of the python file included, and it should create a new folder containing all the images.

### for_training:

train.py which trains for 500 epochs by default
add your data in the **dataset** folder, furry and non-furry respectively
also add some data to **dataset_test** folder for tests to run correctly.

Tests, model and checkpoint saving happen every 5 epochs but
bcan be configured to run after diffrent numbers of epochs in the python files first couple of variables.
