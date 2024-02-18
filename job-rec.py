import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Load the dataset
def load_data():
    return dd.read_csv("merged.csv")

# Preprocess the text data
def preprocess_text(text):
    return text.lower()

# Generate recommendations
def get_recommendations(job_title, data):
    # Filter data based on fuzzy string matching
    filtered_data = data[data['title'].apply(lambda x: fuzz.partial_ratio(x.lower(), job_title.lower()) > 80)]
    if filtered_data.shape[0] == 0:
        return dd.DataFrame()
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_data['title'])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the job title
    idx = filtered_data.index[0]

    # Get the pairwise similarity scores with other jobs
    sim_scores = cosine_sim[idx]

    # Sort the jobs based on the similarity scores
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar jobs
    sim_scores = sim_scores[1:11]  # Exclude the first item (self)

    # Get the job indices
    job_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended job titles
    return filtered_data['title'].iloc[job_indices]

# Main function
def main():
    # Load data
    data = load_data()

    # User input for job title
    job_title = input("Enter the job title: ")

    # Get recommendations
    recommendations = get_recommendations(job_title, data)
    print("Recommended jobs:")
    print(recommendations)

if __name__ == "__main__":
    main()
