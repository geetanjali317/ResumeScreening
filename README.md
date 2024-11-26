# ResumeScreening
Here's an updated description of the Resume Screening system, incorporating Random Forest, SVC, and KNN models instead of Naive Bayes.
Project Steps Overview

    Import Libraries
    Import the necessary libraries for data manipulation, machine learning, and model evaluation.

    Load the Dataset
    Load the dataset containing resumes and their corresponding categories for training.

    Exploring Categories
    Explore the different categories of the resumes (e.g., job titles or skills) and visualize their distribution.

    Exploring Resume
    Take a look at a sample resume to understand the structure and content of the data.

    Balance Classes (Categories)
    If the dataset is imbalanced, apply techniques like SMOTE to balance the number of categories for training the model.

    Cleaning Data
    Preprocess the text data by cleaning it—removing special characters, numbers, and extra spaces. The cleaned text will be used for feature extraction.

    Words into Categorical Values
    Convert categorical labels (e.g., job categories) into numerical values using label encoding to make them suitable for machine learning algorithms.

    Vectorization
    Convert the cleaned text data (resumes) into numerical features using TF-IDF vectorization, which helps capture the importance of words in the context of resumes.

    Train Test Split
    Split the dataset into training and testing sets to evaluate the performance of the model.

    Model Training and Evaluation
    Train models using Random Forest, SVC, and KNN classifiers. Evaluate their performance using metrics like accuracy, precision, recall, and F1-score.

    Prediction System
    Build a function that predicts the category of a new resume based on the trained model.

    Save the Model
    Save the trained models and vectorizer for future predictions or deployment.

Model Comparison

    Random Forest: A powerful ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
    Support Vector Classifier (SVC): A classification model that works well with high-dimensional datasets like text data.
    K-Nearest Neighbors (KNN): A simple and effective classification method that assigns a category based on the majority class among the nearest neighbors.

Final Model

Once trained, the model can predict whether a resume matches a particular category or job title based on the input features. It can also be used in a web interface or deployed as a microservice to assist HR teams in the hiring process.

Let me know if you'd like further guidance on implementing or fine-tuning these models!
