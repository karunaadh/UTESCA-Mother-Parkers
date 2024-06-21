Coffee Bean Classification with Convolutional Neural Network (CNN) and Res-Net
This project aims to classify images of coffee beans using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The CNN is trained on a dataset of coffee bean images to classify them into different categories.

Prerequisites
Before running the code, ensure you have the following dependencies installed:

TensorFlow
OpenCV (cv2)
Matplotlib
NumPy

!!! If you are trying to run this one your machine, and use your GPU, pay close atention to this link: https://www.tensorflow.org/install/pip
The page explains how to properly install tenserflow, CUDA, and cuDNN, along with NVIDIA drivers just so that tenserflow will see your GPU in the notebook. Pay attention to versions of python, tenserflow, CUDA, and cuDNN. If some versions are not compatible, tenserflow will not see your GPU

The first notebook, model1, loads the dataset, preprocesses the images, builds a CNN model, compiles the model with an Adam optimizer, and trains the model on the dataset. The final, hyperparameter-adjusted version is displayed. After training the model, it is evaluated on a separate test set to assess its performance. The evaluation metrics include precision, recall, and accuracy, along with a few others for your interest.

The second notebook, uses a different dataset, with 4 labels, attempting to classify them using the previous CNN method, and also Res-Net, a more complex model with a better final performance.

Saving the Model
Once trained, the model is saved in the models/ directory as CoffeeModel for future use or deployment. The second model's best weights are also saved in the weights folder.

The second model presented will probably produce hectic validation metrics, jumping from high validation accuracy to low, signifying a potential overfit. The reason for that is the limited dataset, which does not allow a Residual network to be as effective as it can be. Yet, during training, the model would reach satistfying validation metrics, hence why I decided to save the best weights. If needed, you can always load those values from the weights folder.

DATA:
#1. https://github.com/tanius/smallopticalsorter/tree/master/classifier-trainingdata <---- Dataset with two labels, good and bad
#2. https://comvis.unsyiah.ac.id/usk-coffee/ <--- A dataset with 4 labels, used in the 2nd Notebook

Author - Ivan Kraskov
