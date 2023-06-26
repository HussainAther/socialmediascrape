from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function for evaluating models using cross-validation
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    avg_accuracy = scores.mean()
    precision = cross_val_score(model, X, y, cv=5, scoring='precision_macro').mean()
    recall = cross_val_score(model, X, y, cv=5, scoring='recall_macro').mean()
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1_macro').mean()
    return avg_accuracy, precision, recall, f1

# Evaluate Naive Bayes model
nb_avg_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_model(nb_model, word2vec_matrix, tweets_df['Sentiment'])

# Evaluate SVM model
svm_avg_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, word2vec_matrix, tweets_df['Sentiment'])

# Evaluate Random Forest model
rf_avg_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, word2vec_matrix, tweets_df['Sentiment'])

# Evaluate RNN model
rnn_avg_accuracy, rnn_precision, rnn_recall, rnn_f1 = evaluate_model(rnn_model, word2vec_matrix, tweets_df['Sentiment'])

# Print evaluation metrics
print("Evaluation Metrics for Naive Bayes:")
print("Average Accuracy:", nb_avg_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1-Score:", nb_f1)

print("\nEvaluation Metrics for SVM:")
print("Average Accuracy:", svm_avg_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-Score:", svm_f1)

print("\nEvaluation Metrics for Random Forest:")
print("Average Accuracy:", rf_avg_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-Score:", rf_f1)

print("\nEvaluation Metrics for RNN:")
print("Average Accuracy:", rnn_avg_accuracy)
print("Precision:", rnn_precision)
print("Recall:", rnn_recall)
print("F1-Score:", rnn_f1)

