# MNIST Classification using Random Forest

This project is focused on building a machine-learning model to classify handwritten digits using the MNIST dataset. The primary objective is to create a classification model using the `RandomForestClassifier` from the `sklearn` library and to evaluate its performance on test data.

### Project Structure

* `mnist_task.ipynb`: The Jupyter Notebook contains the entire code for loading the dataset, preprocessing, training the model, and evaluating the results.
* `README.md`: This README file, which provides an overview of the project, tools used, and evaluation metrics.

### Dataset
The MNIST dataset contains 70,000 images of handwritten digits (0-9) in grayscale. Each image is 28x28 pixels, and the dataset is often used for training various image processing systems. However, in this notebook, we use a smaller subset from `sklearn.datasets`, where the images are 8x8 pixels.

### Requirements
To run this notebook, you need the following Python libraries:

* `numpy`
* `matplotlib`
* `sklearn`

You can install them using pip:

* bash (Copy code)
  * pip install numpy matplotlib scikit-learn

### Analysis and Model Training

#### Loading the Dataset
The MNIST dataset is loaded using `load_digits()` from sklearn.datasets.

#### Data Splitting
The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`. This is crucial for evaluating the model's performance on unseen data.

#### Model Building
A Random Forest model is built using `RandomForestClassifier` from `sklearn.ensemble`. We tune the parameters `n_estimators` (number of trees) and `max_depth` (maximum depth of each tree) to find an optimal balance between bias and variance.

#### Model Evaluation
The model is evaluated using a confusion matrix and classification report. Key metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance.

### Results
* Accuracy: 97.2%
* Precision: 97.4%
* Recall: 97.3%
* F1-score: 97.3%

The model performs well overall but struggles with distinguishing between certain similar classes, such as 1 and 9.

### Tools and Libraries
* Scikit-learn: For loading the dataset, model creation, and evaluation.
* Matplotlib: For visualizing the digits in the dataset.
* NumPy: For numerical operations.

### EDA and Visualisation
Basic exploratory data analysis (EDA) is performed by visualising a few samples from the dataset. Each digit is displayed as an 8x8 pixel image, providing an understanding of the data's structure.

### Future Work
* Experimenting with different classifiers (e.g., SVM, CNN) for potentially better results.
* Implementing hyperparameter tuning methods such as GridSearchCV for optimal parameter selection.
* Adding more EDA and visualisations to understand the dataset better.

----------------------------------------------------------------------------------------

#### References:
https://www.kaggle.com/code/manthansolanki/image-classification-with-mnist-dataset

https://machinelearningmastery.com/difference-test-validation-datasets/

https://www.w3schools.com/python/python_ml_train_test.asp

https://www.kaggle.com/code/mithlesh14/mnist-image-processing/notebook

##### Thank you, Farinaaz :)
