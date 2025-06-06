from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV

newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
categories = newsgroups_data.target_names

X = newsgroups_data.data
y = newsgroups_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

count_vectorizer = CountVectorizer()
tf_vectorizer = CountVectorizer(binary=True)
tfidf_vectorizer = TfidfVectorizer()

X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

X_train_tf = tf_vectorizer.fit_transform(X_train)
X_test_tf = tf_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

def evaluate_model(classifier, X_train, X_test, y_train, y_test, description):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"\nResults for {description}:")
    print(classification_report(y_test, y_pred, target_names=categories))


nb_classifier = MultinomialNB()
evaluate_model(nb_classifier, X_train_counts, X_test_counts, y_train, y_test, description="Naive Bayes with Count Vectorizer")

evaluate_model(nb_classifier, X_train_tf, X_test_tf, y_train, y_test, description="Naive Bayes with Term Frequency Vectorizer")

evaluate_model(nb_classifier, X_train_tfidf, X_test_tfidf, y_train, y_test, description="Naive Bayes with TF-IDF Vectorizer")

lr_classifier = LogisticRegression(max_iter=1000)
evaluate_model(lr_classifier, X_train_counts, X_test_counts, y_train, y_test, description="Logistic Regression with Count Vectorizer")

evaluate_model(lr_classifier, X_train_tf, X_test_tf, y_train, y_test, description="Logistic Regression with Term Frequency Vectorizer")

evaluate_model(lr_classifier, X_train_tfidf, X_test_tfidf, y_train, y_test, description="Logistic Regression with TF-IDF Vectorizer")

svm_classifier = SVC(kernel='linear')
evaluate_model(svm_classifier, X_train_counts, X_test_counts, y_train, y_test, description="SVM with Count Vectorizer")

evaluate_model(svm_classifier, X_train_tf, X_test_tf, y_train, y_test, description="SVM with Term Frequency Vectorizer")

evaluate_model(svm_classifier, X_train_tfidf, X_test_tfidf, y_train, y_test, description="SVM with TF-IDF Vectorizer")


tfidf_vectorizer = TfidfVectorizer(lowercase=True)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear')
evaluate_model(svm_classifier, X_train_tfidf, X_test_tfidf, y_train, y_test, description="SVM with TF-IDF (lowercase=True)")

tfidf_vectorizer_no_lower = TfidfVectorizer(lowercase=False)

X_train_tfidf_no_lower = tfidf_vectorizer_no_lower.fit_transform(X_train)
X_test_tfidf_no_lower = tfidf_vectorizer_no_lower.transform(X_test)

evaluate_model(svm_classifier, X_train_tfidf_no_lower, X_test_tfidf_no_lower, y_train, y_test, description="SVM with TF-IDF (lowercase=False)")

tfidf_vectorizer_stopwords = TfidfVectorizer(stop_words='english')


X_train_tfidf_stopwords = tfidf_vectorizer_stopwords.fit_transform(X_train)
X_test_tfidf_stopwords = tfidf_vectorizer_stopwords.transform(X_test)

evaluate_model(svm_classifier, X_train_tfidf_stopwords, X_test_tfidf_stopwords, y_train, y_test, description="SVM with TF-IDF (stop_words='english')")


tfidf_vectorizer_bigrams = TfidfVectorizer(ngram_range=(1, 2))

X_train_tfidf_bigrams = tfidf_vectorizer_bigrams.fit_transform(X_train)
X_test_tfidf_bigrams = tfidf_vectorizer_bigrams.transform(X_test)

evaluate_model(svm_classifier, X_train_tfidf_bigrams, X_test_tfidf_bigrams, y_train, y_test, description="SVM with TF-IDF (ngram_range=(1,2))")


tfidf_vectorizer_max_features = TfidfVectorizer(max_features=5000)


X_train_tfidf_max_features = tfidf_vectorizer_max_features.fit_transform(X_train)
X_test_tfidf_max_features = tfidf_vectorizer_max_features.transform(X_test)

evaluate_model(svm_classifier, X_train_tfidf_max_features, X_test_tfidf_max_features, y_train, y_test, description="SVM with TF-IDF (max_features=5000)")