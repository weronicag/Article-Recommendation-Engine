from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    name = [(a[0],a[1]) for a in articles]
    return render_template('articles.html', articles=name)


@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    article_idx = [elm[0] for elm in articles].index(topic + '/' + filename)
    article = articles[article_idx]
    key = article[0]
    title = article[1]
    text = article[2] 
    recommended = seealso[key]
    #send input to html/css formating 
    return render_template('article.html', text=text, title=title, recommended=recommended)


i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)
seealso = all_recommended(articles, n=5)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
