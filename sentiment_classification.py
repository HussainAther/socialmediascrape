from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(word2vec_matrix, tweets_df['Sentiment'], test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Train Support Vector Machines classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Train Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Train Recurrent Neural Network (RNN) classifier
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=word2vec_matrix.shape[0], output_dim=100, input_length=word2vec_matrix.shape[1]))
rnn_model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
rnn_model.add(Dense(units=1, activation='sigmoid'))
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(word2vec_matrix, tweets_df['Sentiment'], epochs=10, batch_size=32, validation_split=0.2)
rnn_predictions = rnn_model.predict_classes(X_test)
rnn_accuracy = accuracy_score(y_test, rnn_predictions)

# Evaluate the models
print("Naive Bayes Accuracy:", nb_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("RNN Accuracy:", rnn_accuracy)

print("\nClassification Report for RNN:")
print(classification_report(y_test, rnn_predictions))

