CS-455 Final Project | Road Lane and Traffic Sign Detection

Sharif Adepetu, Amanda Wills, Andrew Zubyk
-------------------------------------------------------------

Welcome to the Road Lane and Traffic Sign Detection project! This repository contains code for building and training Convolutional Neural Networks (CNNs) to detect road lanes and recognize traffic signs. The project combines traditional computer vision techniques with machine learning to provide a comprehensive solution for autonomous driving applications.

Requirements

To run this project, you need Python 3.6 or higher and the following packages:

TensorFlow
OpenCV
NumPy
Matplotlib
PIL (Python Imaging Library)
You can find the complete list of required packages in the requirements.txt file.

Installation

Clone the repository:
bash
Copy code
git clone <your-repository-url>
cd <your-repository-folder>
Set up the Python environment:
You can use a virtual environment (e.g., venv, conda) to isolate the project dependencies from your system Python environment.
bash
Copy code
# If using venv
python -m venv myenv
source myenv/bin/activate  # For Linux/MacOS
myenv\Scripts\activate     # For Windows
Install the required packages:
After activating your virtual environment, run the following command to install the dependencies listed in requirements.txt:
bash
Copy code
pip install -r requirements.txt
Data Preparation

To train the model for traffic sign recognition and road lane detection, ensure you have the necessary datasets in the correct directories.

Traffic Sign Recognition:
Organize your traffic sign images into train, validation, and test directories within a parent directory (sign_data/traffic_signs).
Ensure each subdirectory contains the appropriate classes, such as Light and Stop.
Road Lane Detection:
Place road lane images in data/train, data/validation, and data/test directories.
Running the Project

Once you have installed the required packages and prepared the data, you can run the code in the following order:

Package Installation:
Run package_installer.py to ensure that all necessary packages are installed:
bash
Copy code
python package_installer.py
Traffic Sign Recognition:
Run signprediction.py to build, train, and evaluate a CNN for traffic sign recognition:
bash
Copy code
python signprediction.py
This script will save the trained model and output predictions for unknown traffic sign images.
Road Lane Detection:
Run main.py to perform road lane detection using OpenCV:
bash
Copy code
python main.py
This script will process static road images to detect and visualize lane boundaries.
Additional Tasks:
To build, train, and evaluate a model for road lane detection, run project.py:
bash
Copy code
python project.py
This script includes data generators, model training, validation, and visualization of correct and incorrect predictions.
Notes

Ensure that the correct dataset structure is maintained for traffic sign recognition and road lane detection.
Early stopping is implemented in the training process to prevent overfitting.
The scripts output visualizations to help understand the results, such as plots showing training and validation accuracy and images displaying detected lanes or traffic sign predictions.
Contribution and Issues

Feel free to contribute to this project by creating a pull request or opening an issue if you encounter any problems. Your feedback and contributions are highly appreciated!


From the Project Instructions:

Your repository should provide a read me with clear instructions on how to compile and execute your work.
Build files such as makefiles, gradle files, script, etc.. are appreciated when applicable.
It must also provide clear instructions on any required resources and instructions (or links to instructions) to install those resources.

Be sure to cite within your comments any use of external code regardless of the source's license.

Note: Use of 3rd party developed source code (including code snippets borrowed from Stack Overflow) are properly cited in the code with a comment and/or other appropriate documentation as per the source's license.
