Dataset Description:

The dataset contains the following columns:

v1: The label indicating whether the message is 'ham' (non-spam) or 'spam'.
v2: The text of the message.




Spam Classifier Model:

This project implements a machine learning model to classify SMS messages as either spam or ham (non-spam). The model is built using a Random Forest classifier and TF-IDF vectorization for text feature extraction. To use this project, you will need Python 3.x and the following libraries: pandas, spacy, scikit-learn, joblib, matplotlib (for optional visualization), and wordcloud (for optional visualization). You can install these packages using pip. Additionally, you will need to download spaCy's English model.

First, clone the repository and navigate to the project directory. Then, install the required packages and download the spaCy English model. The dataset spam.csv should be in the same directory as the notebook or script. The preprocessing steps include removing unnecessary columns, encoding labels, and cleaning the text data. The model training involves splitting the dataset into training and testing sets and training a Random Forest classifier on the training data. The model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score. After training, the model and the TF-IDF vectorizer are saved using joblib.

To predict user input, load the saved model and vectorizer, preprocess the user input, and classify it as spam or ham. An example script demonstrating the entire process is provided in the README, including loading the dataset, preprocessing the text, vectorizing using TF-IDF, training the Random Forest classifier, evaluating the model, saving the model and vectorizer, and predicting user input. Ensure the dataset spam.csv is present in the working directory and adjust the file paths as necessary. For more detailed steps, refer to the provided Jupyter notebook Spam_Classifier.ipynb. 
