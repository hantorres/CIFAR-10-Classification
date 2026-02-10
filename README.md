# CIFAR-10 Classification
Building a CNN to classify the labels of the CIFAR-10 dataset.

## Data Cleaning - Python
### Method
The data is first loaded into memory using Pandas. The Os library is then used to create new columns in the pandas dataframe to add the filepath to the corresponding image and encode the image labels using label encoding.
A TensorFlow dataset is then created using the image filepath and its numerical label.
Image size is first declared, then a loading and cleaning function is initialized to process the data. It takes the following steps: read the image from the file path, scanning for null/erroneous files. Initialize the image resolution and convert it to an array of floats for processing. Identify unusual characters and occlusions by scanning image for outliers via z-score. Reduce noise by increasing contrast and brightness. Remove blur through convolution with a sharpening filter. Resize, pad, and normalize the image after processing.
An augmentation function is then declared, where the following modifications are made on the images: random vertical flip, horizontal flip, and saturation.
After declaring these functions, run the dataset through these processes and remove any instances where data processing fails. 
After processing the data, create data subsets for training, validation, and testing.
Lastly, serialize the samples and then write them to a TFRecord file, compressed with GZIP for storage optimization. 

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Pandas
2) Os
3) TensorFlow
4) Numpy
5) Matplotlib

The notebook was tested using Python 3.9.6.

### Pre-requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

    
Note: Dataset and the specified directories within the notebook are NOT included in this repository due to file size concerns; however, a zipped version of the test dataset is included. This dataset has been locally sourced and includes both a folder of raw images and a .csv file containing the labels for each image. The dataset follows the same standards/configurations as other CIFAR-10 sources.
An integrated development environment (IDE) is necessary to run this notebook. 

### Version History
V1.0 - The Jupyter Notebook is created. Functionality of all cells and functions has been tested and is functional.


## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook. Furthermore, the outputs of the program are the following:
1) Processed dataset splits: training, validation, and test datasets are saved. These datasets are all cleaned, labeled, and compressed when saved onto local machine.


Note: Program outputs are intended to be used in scripts/notebooks external to this project.

### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset.

Using the notebook:
1) Open the notebook within IDE.
2) Run all cells in the notebook.
3) Validate process externally by checking if the dataset splits were saved locally.
4) The dataset is now ready to be used in another project.

Note: Writing the dataset files can take ~2 minutes.

Uploading the files in a new project is fairly simple. It requires a defined dictionary to store the data in, a parser to read each sample and fit it into the format of the dictionary, and then a variable to save the loaded dataset to. An example to loading the train data into a new project is the following:

```python
import tensorflow as tf

# Format in which the dataset will be loaded in
feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

# Parser to read each sample and put it into the format of the dictionary
def parse_sample(sample):
    parsed = tf.io.parse_single_example(sample, feature_desc)
    
    img = tf.io.parse_tensor(parsed["image"], out_type=tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    
    label = parsed["label"]
    return img, label

# Loading in the trained dataset
train_ds = (
    tf.data.TFRecordDataset("train.tfrecord.gz", compression_type="GZIP")
    .map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
)
```

## Model Development - Python
### Method
The processed data is first loaded into memory as a TensorFlow dataset object.
A data augmentation process is defined using the Keras API to perform the following operations: resizing, random horizontal flipping, random rotation, random contrast, random zooming, and random translation.
The convolutional neural network is built with the Keras API, including 6 convolutional layers and 2 fully connected layers. Convolutional layers use a ReLU activation, and a SoftMax activation is used for classification. 
Batch normalization and dropout layers are employed to prevent overfitting. The model uses max pooling to pool the data. 
The model is trained using the Adam optimizer and a sparse categorical cross-entropy loss function. The model has 2.6 million trainable parameters.
The model is trained using early stopping to prevent overfitting with a batch size of 64.
During model training, model evaluation is performed by visualizing the loss and accuracy curves and a confusion matrix of predicted labels. 
Lastly, save the model to local storage for external use. 

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) TensorFlow
2) Keras
3) Matplotlib
4) Numpy
5) Seaborn

The notebook was tested using Python 3.9.6.

### Pre-requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

    
Note: Dataset and the specified directories within the notebook are NOT included in this repository due to file size concerns; however, a zipped version of the test dataset is included. The final model parameters are included in the repository for use. 
An integrated development environment (IDE) is necessary to run this notebook. 

### Version History
V1.0 - The Jupyter Notebook is created. Functionality of all cells and functions has been tested and is functional.


## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook. Furthermore, the outputs of the program are the following:
1) Keras model object containing model parameters.
2) Model evaluation figures: Loss Curve, Accuracy Curve, and Confusion Matrix.
3) Training history: a .csv containing the evolution of accuracy and loss with respect to epoch.

### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset.

Using the notebook:
1) Open the notebook within IDE.
2) Run all cells in the notebook.
3) Validate process externally by checking if the model parameters were saved locally and figures are logically sound.
4) The model is now ready to be used/deployed.
