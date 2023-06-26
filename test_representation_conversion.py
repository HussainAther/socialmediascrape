from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Create an instance of CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the preprocessed text data using CountVectorizer
count_matrix = count_vectorizer.fit_transform(tweets_df['Cleaned Tweet'])

# Create an instance of TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed text data using TfidfVectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets_df['Cleaned Tweet'])

# Train Word2Vec model on the preprocessed text data
tokenized_tweets = [tweet.split() for tweet in tweets_df['Cleaned Tweet']]
word2vec_model = Word2Vec(tokenized_tweets, size=100, window=5, min_count=1)

# Function to convert a single preprocessed tweet to its Word2Vec representation
def get_tweet_vector(tweet):
    vectors = []
    for word in tweet.split():
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    if len(vectors) > 0:
        tweet_vector = np.mean(vectors, axis=0)
    else:
        tweet_vector = np.zeros(100)
    return tweet_vector

# Convert each preprocessed tweet to its Word2Vec representation
word2vec_matrix = np.vstack([get_tweet_vector(tweet) for tweet in tweets_df['Cleaned Tweet']])

# Print the shape of the numerical representations
print("Bag-of-Words matrix shape:", count_matrix.shape)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Word2Vec matrix shape:", word2vec_matrix.shape)

