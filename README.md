# Job Recommendation System

This is a job recommendation system built using Streamlit. The system allows users to input their job preferences and location, and it recommends jobs that match their preferences.

## Installation

To run the app locally, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone <repository_url>
    ```

2. Navigate to the project directory:

    ```bash
    cd job-recommendation-system
    ```

3. Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run job-reco.py
    ```

## Usage

1. Upon running the app, you will see a text input field where you can enter your job preference.
2. Enter your preferred job and location in the respective fields.
3. Click on the "Recommend" button to view recommended jobs based on your preferences.
4. The recommended jobs will be displayed below the input fields.

## Data

The app loads job data from a CSV file stored on Google Drive. The data is preprocessed to filter out duplicate job IDs, convert job IDs to integers, and combine relevant text features for recommendation.

## Files

- '1_EDA.ipynb': Contains Task 1 (Data Analysis)
- '2A_use_cases.ipynb': Contains Task 2A (Use cases for this dataset)
- `job-reco.py`: Contains the main code for the Streamlit app.
- `requirements.txt`: Specifies the required Python packages for the app.

## Credits

- This app was created by Muhammad 'Afif Amir Husin.

