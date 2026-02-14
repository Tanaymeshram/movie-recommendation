from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return ["Movie not found"]

    index = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
