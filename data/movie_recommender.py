import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

class MovieRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.movie_data = pd.read_csv(self.data_path)
        self.selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        self.vectorizer = TfidfVectorizer()
        self.feature_vectors = None
        self.similarity_matrix = None
        self.load_and_process_data()

    def load_and_process_data(self):
        for feature in self.selected_features:
            self.movie_data[feature] = self.movie_data[feature].fillna('')

        combined_features = self.movie_data['genres'] + ' ' + self.movie_data['keywords'] + ' ' + \
                            self.movie_data['tagline'] + ' ' + self.movie_data['cast'] + ' ' + \
                            self.movie_data['director']
        
        self.feature_vectors = self.vectorizer.fit_transform(combined_features)
        self.similarity_matrix = cosine_similarity(self.feature_vectors)

    def recommend_movies(self, movie_name, num_recommendations=30):
        list_of_all_titles = self.movie_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if not find_close_match:
            print("No close matches found for the provided movie name.")
            return []

        close_match = find_close_match[0]
        index_of_the_movie = self.movie_data[self.movie_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(self.similarity_matrix[index_of_the_movie]))
        sorted_similar_movie = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movie):
            index = movie[0]
            title_from_index = self.movie_data[self.movie_data.index == index]['title'].values[0]
            if i < num_recommendations:
                recommended_movies.append(title_from_index)

        return recommended_movies
