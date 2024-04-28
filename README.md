CS-455 Final Project | Road Lane and Traffic Sign Detection
-------------------------------------------------------------

Welcome to the Road Lane and Traffic Sign Detection project! This repository contains code for building and training Convolutional Neural Networks (CNNs) to detect road lanes and recognize traffic signs. The project combines traditional computer vision techniques with machine learning to provide a comprehensive solution for autonomous driving applications.

Requirements:

To run this project, you need Python 3.6 or higher and the following packages:

TensorFlow,
OpenCV,
NumPy,
Matplotlib,
PIL (Python Imaging Library),
You can find the complete list of required packages in the requirements.txt file.

Installation:

1. Set up the Python environment:
  You can use a virtual environment (e.g., venv, conda) to isolate the project dependencies from your system   Python environment.

2. Install the required packages:
  After activating your virtual environment, run the following command to install the dependencies listed in   requirements.txt:


Data Preparation:

To train the model for traffic sign recognition and road lane detection, ensure you have the necessary datasets in the correct directories.

1. Traffic Sign Recognition:
  Organize your traffic sign images into train, validation, and test directories within a parent directory       (sign_data/traffic_signs).
  Ensure each subdirectory contains the appropriate classes, such as Light and Stop.
2. Road Lane Detection:
  Place road lane images in data/train, data/validation, and data/test directories.


Running the Project:

Once you have installed the required packages and prepared the data, you can run the code in the following order:

1. Package Installation:
  Run package_installer.py to ensure that all necessary packages are installed:
    python package_installer.py
2. Traffic Sign Recognition:
  Run signprediction.py to build, train, and evaluate a CNN for traffic sign recognition:
    python signprediction.py
  This script will save the trained model and output predictions for unknown traffic sign images.
3. Road Lane Detection:
  Run main.py to perform road lane detection using OpenCV:
    python main.py
  This script will process static road images to detect and visualize lane boundaries.
4. Additional Tasks:
  To build, train, and evaluate a model for road lane detection, run project.py:
    python project.py
  This script includes data generators, model training, validation, and visualization of correct and       incorrect predictions.


Notes:

  Ensure that the correct dataset structure is maintained for traffic sign recognition and road lane detection.
  Early stopping is implemented in the training process to prevent overfitting.
  The scripts output visualizations to help understand the results, such as plots showing training and validation accuracy and images displaying detected lanes or traffic sign predictions.


Contribution and Issues:

Feel free to contribute to this project by creating a pull request or opening an issue if you encounter any problems. Your feedback and contributions are highly appreciated!

Be sure to cite within your comments any use of external code regardless of the source's license.
