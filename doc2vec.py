import sys
import re
import string
import os
import numpy as np
import codecs


ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Reads all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:
    the 0.418 0.24968 -0.41242 0.1217 ...
    """
    print('load_glove processing...')
    with open(filename, 'r') as f:
        word_vect_dict = {}
        for line in f.readlines():
            line = line.strip()
            elements = line.split(' ')
            word = elements[0]
            vector = np.array(elements[1:], dtype=float)  # remove
            word_vect_dict[word] = vector
    return word_vect_dict


def filelist(root):
    """Returns a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name == 'COPYRIGHT':
                continue
            allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Loads and returns the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text):
    """
    Given a string, returns a list of normalized words.
    """
    text_lower = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text_lower)
    word_list = text.split(' ')
    word_list = [w for w in word_list if len(w) > 2]
    return [word for word in word_list if word not in ENGLISH_STOP_WORDS]


def load_articles(articles_dirname, gloves):
    """
    Loads all .txt files under articles_dirname and return a table (list of lists/tuples)
    where each record is a list of:
      [filename, title, article-text-minus-title, wordvec-centroid-for-article-text]
    Uses gloves parameter to compute the word vectors and centroid.
    """
    print('load_articles processing...')
    output_table = []
    full_filepath = filelist(articles_dirname)
    for file_path in full_filepath:
        filename = '/'.join(file_path.split('/')[-2:])
        text = get_text(file_path)
        article_title = text.split('\n')[0]
        article_text = ','.join(text.split('\n')[1:])
        centroid = doc2vec(text, gloves)
        output_table.append((filename, article_title, article_text, centroid))
    return output_table


def doc2vec(text, gloves):
    """
    Returns the word vector centroid for the text. Sums the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """
    # given a string of text
    word_list = words(text)
    sum_vectors = np.zeros(len(gloves['the']))
    for word in word_list:
        if word not in gloves.keys():
            continue
        sum_vectors = np.add(sum_vectors, gloves[word])
    return np.true_divide(sum_vectors, len(word_list))  # returns the centroid


def distances(article, articles):
    """
    Computes the euclidean distance from article to every other article and return
    a list of (distance, a) tuples for all a in articles.
    """
    distance_articles = []
    for a in articles:
        if article == a:
            continue
        distance = np.linalg.norm(article[3] - a[3])
        distance_articles.append((distance, a))
    return distance_articles


def recommended(article, articles, n=5):
    """
    Returns a list of the n articles (records with filename, title, etc...)
    closest to article's word vector centroid.
    """
    distance_articles = []
    for a in articles:
        if article == a:
            continue
        distance = np.linalg.norm(article[3] - a[3])
        if distance > 0:
            distance_articles.append((distance, a))
    distance_articles.sort(key=lambda x: x[0])
    return distance_articles[:n]


def all_recommended(articles, n=5):
    """
    Returns a dictonary with the top n recommended article
    for each article in articles
    """
    recommender_dict = {}
    for article in articles:
        key = article[0]
        art_recommendations = recommended(article, articles, n)
        if key not in recommender_dict:
            value = [(elm[1][0], elm[1][1]) for elm in art_recommendations]
            recommender_dict[key] = value
    return recommender_dict
