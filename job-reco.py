import streamlit as st
import pandas as pd
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# Load data
@st.cache_data
def load_data(csv_file_url, chunksize):
    gdown.download(csv_file_url, 'data.csv', quiet=False)
    all_chunks = []
    for chunk in pd.read_csv('data.csv', chunksize=chunksize):
        all_chunks.append(chunk)
    return pd.concat(all_chunks, ignore_index=True)

# Preprocess data
def preprocess_data(df):
    # Drop rows with NaN values in title, abstract, or content columns
    df = df.dropna(subset=['title', 'abstract', 'content'])
    
    # Filter out duplicate job IDs
    unique_job_ids = set(df['id'])
    data = df[df['id'].isin(unique_job_ids)]

    # Convert job_id to integers
    data['id'] = pd.to_numeric(data['id'], errors='coerce').fillna(0).astype(int)

    # Drop rows with NaN values in the 'job_id' column
    data = data.dropna(subset=['id'])

    # Combine relevant text features
    data["combined_text"] = data["title"] + " " + data["abstract"] + " " + data["content"]
    data["combined_text"] = data["combined_text"].apply(lambda x: x if isinstance(x, str) else '')

    return data

# Main function
def main():
    st.title('Job Recommendation System')

    # Load data from Google Drive
    csv_file_url = "https://drive.google.com/file/d/1M07tqmbUKqdDf6k6CKjmjB28k5Qjxre2/view?usp=sharing"
    file_id = csv_file_url.split('/')[-2]
    download_link = f"https://drive.google.com/uc?id={file_id}"
    
    chunksize = 20000
    data = load_data(download_link, chunksize)

    # Preprocess data
    data = preprocess_data(data)

    # User input
    user_query = st.text_input('Enter your job preference:', 'assistant')
    user_location = st.text_input('Enter your location:', 'New York')

    if st.button('Recommend'):
        # Clean and preprocess text data
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform(data["combined_text"])

        # One-hot encode location data
        location_encoder = OneHotEncoder(handle_unknown='ignore')
        location_features = location_encoder.fit_transform(data[['location_name']])

        # Combine TF-IDF vectors and location features
        combined_features = hstack([tfidf_matrix, location_features])

        # Calculate cosine similarity
        user_tfidf = tfidf_vectorizer.transform([user_query])
        user_location_encoded = location_encoder.transform([[user_location]])
        user_features = hstack([user_tfidf, user_location_encoded])
        cosine_similarities = cosine_similarity(user_features, combined_features)

        # Get top-5 recommended jobs
        N = 5
        similar_jobs_indices = cosine_similarities.argsort()[0][-N:][::-1]
        recommended_jobs_details = data.iloc[similar_jobs_indices][["id", "title", "abstract", "content", "location_name"]]

        # Filter out duplicate job IDs
        recommended_jobs_details = recommended_jobs_details.drop_duplicates(subset='id')

        # Display recommended jobs
        st.subheader('Recommended Jobs:')
        st.write(recommended_jobs_details)


if __name__ == "__main__":
    main()
