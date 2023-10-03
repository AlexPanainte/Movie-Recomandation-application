from data import movie_recommender as mr

def main():
    data_path = "data/movies.csv"  # Specifică calea către fișierul CSV cu date
    recommender =mr.MovieRecommender(data_path)

    while True:
        movie_name = input('Enter your favorite movie name (or "exit" to quit): ')
        if movie_name.lower() == "exit":
            break

        recommended_movies = recommender.recommend_movies(movie_name)
        
        if recommended_movies:
            print('\nMovies suggested for you:\n')
            for i, movie in enumerate(recommended_movies, start=1):
                print(f'{i}. {movie}')
        else:
            print("No recommendations found.")

if __name__ == "__main__":
    main()
