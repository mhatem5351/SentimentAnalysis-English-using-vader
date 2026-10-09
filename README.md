# Sentiment analysis of movie reviews with VADER

Text cleaning, word-frequency analysis and VADER sentiment scoring on 1,000 IMDB movie reviews, followed by a
comparison of classical classifiers trained on TF-IDF features.

This is an assignment for the NLP course of the ITI AI-Pro diploma (2021–22).

> **Note:** The Decision Tree's 100% test accuracy in the notebook is data leakage. It was fit on the full dataset
> (`fit(X, Y)`) and then scored on a test set it had already seen. That number is not a real result.

## Data

`TextAnalytics.txt` (1.3 MB): 1,000 IMDB movie reviews, provided with the assignment. They match the first rows of the
Kaggle [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The file has no sentiment labels.

## Approach

1. Clean the text: remove digits, punctuation and `<br />` tags, lower-case it, and lemmatise it. Stop words are
   removed for the word counts and the word cloud.
2. Explore the words with a word cloud and the top-10 word counts ("movie" 2,019, "film" 1,773, ...).
3. Score each review with VADER and turn the compound score into a label: ≥ 0.5 positive, 0 to 0.5 neutral,
   ≤ 0 negative. This gives 568 positive, 369 negative and 63 neutral reviews.
4. Vectorise the text with TF-IDF, split 80/20, and train Logistic Regression, Decision Tree, Random Forest, KNN and a
   linear SVM to predict those labels.

## Results (test set, 200 reviews)

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Linear SVM | 0.944 | 0.695 |
| Logistic Regression | 0.918 | 0.650 |
| Random Forest | 1.000 | 0.605 |
| KNN | 0.713 | 0.550 |
| Decision Tree | 1.000 | 1.000 (leakage, see the note above) |

The models overfit: train accuracy is well above test accuracy. Random Forest, KNN and SVM get none of the 10 neutral
test reviews right.

## How to run

The notebook imports two local helper modules, `nlp_utils` and `contractions` (for `CONTRACTION_MAP`).
They are not in this repository, so the notebook does not run as it is. The other dependencies are pandas, numpy,
NLTK, vaderSentiment, scikit-learn, wordcloud and matplotlib.

## Notes

- The labels come from VADER itself, so the classifiers learn to copy VADER's score, not human-judged sentiment.
  The IMDB dataset's own positive/negative labels would make a better target.
- Removing the string `br` with a regex also removed it from inside words ("brutality" became "utality").
- The notebook is 22.8 MB because it stores large text outputs, so GitHub can be slow to show it.
