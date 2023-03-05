# Article Recommendation Engine

Article recommendation web server that displays a list of BBC articles and related article recommendations. 
This is a simple article recommendation engine that uses word2vec and vector representations for words from Stanford's GloVe project.  

More information on the Stanford GloVe project can be found here: https://nlp.stanford.edu/projects/glove/


Each word has a vector of 300 floating-point numbers in the vector space. Words are related if their word vectors are close in this vector space (these GloVe vectors are based on word-word co-occurrence probabilities derived from a neural network). For example, the word vectors for king and queen live close to each other as they often appear in the same context. If I compute the centroid (sum of the word vectors in the article divided by the number of words in the article) of documents of word vectors, related articles have centroids close in this vector space.

Here is an example of a recommendation from the resulting flask web application:
![article and recommendation example](https://github.com/weronicag/usf-projects/blob/main/article_recommendation_sys/results/article_rec_example.png)

Further improvements to this recommendation engine may factor in collaborative and content-based filtering approaches. 