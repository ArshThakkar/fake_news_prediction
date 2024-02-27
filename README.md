
# Fake News Prediction

This project aims to classify news articles as either real or fake using machine learning techniques. The model is trained on a dataset containing labeled news articles and utilizes Natural Language Processing (NLP) techniques for feature extraction and logistic regression for classification.

## Dataset

The dataset used in this project is stored in a CSV file named `train.csv`. It consists of two columns: `author` and `title`, along with a label column `label` indicating whether the news is real or fake.

## Dependencies

- numpy
- pandas
- re
- nltk
- scikit-learn

You can install the required dependencies using the following command:

```bash
pip install numpy pandas nltk scikit-learn
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-prediction.git
```

2. Navigate to the project directory:

```bash
cd fake-news-prediction
```

3. Run the Jupyter Notebook `fake_news_prediction.ipynb` to train the model and evaluate its performance.

## Instructions

1. Ensure you have Python and Jupyter Notebook installed.
2. Open the Jupyter Notebook `fake_news_prediction.ipynb`.
3. Execute each cell in the notebook sequentially to load the dataset, preprocess the data, train the model, and evaluate its accuracy.
4. After training the model, you can use it to predict whether a news article is real or fake by providing new textual data.

## Usage

To use the trained model for prediction:

1. Provide the news content as input.
2. The model will predict whether the news is real or fake.
3. The predicted label (0 for real, 1 for fake) will be displayed.

## Example

```python
X_new = "New Study Shows Vaccines Are Effective Against COVID-19"
prediction = model.predict(X_new)
if prediction[0] == 0:
    print('The news is real.')
else:
    print('The news is fake.')
```

## Acknowledgments

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/c/fake-news/data).

---
