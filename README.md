Coffee Bean Classification with Convolutional Neural Network (CNN)
This project aims to classify images of coffee beans using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The CNN is trained on a dataset of coffee bean images to classify them into different categories.

Prerequisites
Before running the code, ensure you have the following dependencies installed:

TensorFlow
OpenCV (cv2)
Matplotlib
NumPy

Training the Model
Run the script train_model.py. This script loads the dataset, preprocesses the images, builds a CNN model, compiles the model with an Adam optimizer, and trains the model on the dataset.
Adjust hyperparameters such as learning rate, number of epochs, and model architecture in the script according to your requirements.
Evaluation
After training the model, it is evaluated on a separate test set to assess its performance. The evaluation metrics include precision, recall, and binary accuracy.

Saving the Model
Once trained, the model is saved in the models/ directory as CoffeeModel1.h5 for future use or deployment.

References
TensorFlow: https://www.tensorflow.org/
OpenCV: https://opencv.org/
Matplotlib: https://matplotlib.org/

Author - Ivan Kraskov
