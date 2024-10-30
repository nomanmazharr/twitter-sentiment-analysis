import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import os
import matplotlib.pyplot as plt
import re

# Load environment variables for the API
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api)

# Define function for sentiment analysis
def get_groq_sentiment(text):
    system_prompt = """
    You are an AI agent specializing in sentiment analysis. Your job is to classify the sentiment of each text as positive, neutral, or negative. Please respond with only 'Positive', 'Neutral', or 'Negative' for each text.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=300
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error contacting the GROQ API: {e}")
        return "Error"

# Load dataset from Hugging Face link
def load_dataset(link):
    try:
        if link.endswith(".csv"):
            return pd.read_csv(link)
        else:
            return pd.read_parquet(link)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
def preprocess_tweets(data):
    data['tweet'] = data['tweet'].str.replace(r'@\w+', '', regex=True).str.strip()
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002700-\U000027BF"  
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    data['tweet'] = data['tweet'].str.replace(emoji_pattern, '', regex=True).str.strip()
    data = data[data['tweet'].str.len() > 10]
    data = data.reset_index(drop=True)
    return data

def detect_spikes(data):
    # Group by day or hour to analyze tweet frequency and detect spikes
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    tweet_counts = data.set_index('datetime').resample('D').size()  # Change 'D' to 'H' for hourly analysis
    rolling_mean = tweet_counts.rolling(window=7).mean()
    spike_threshold = rolling_mean.mean() + 2 * rolling_mean.std()  # Adjust threshold as needed
    spikes = tweet_counts[tweet_counts > spike_threshold]
    return tweet_counts, spikes

# Summarize overall sentiment and trend
def analyze_sentiment_trend(data):
    # Ensure datetime is parsed
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    else:
        st.error("The dataset must include a 'datetime' column with timestamp information.")
        return None, None
    
    # Sampling for efficiency
    sample_size = min(50, len(data))
    sample_data = data.sample(n=sample_size, random_state=1)

    # Perform sentiment analysis on each sample tweet
    sample_data['sentiment'] = sample_data['tweet'].apply(get_groq_sentiment)

    # Extract month from datetime for trend analysis
    sample_data['month'] = sample_data['datetime'].dt.to_period('M')

    # Group by month and sentiment
    monthly_trends = sample_data.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

    # Calculate the overall sentiment score for each month
    monthly_sentiment_score = (monthly_trends['Positive'] - monthly_trends['Negative']) / monthly_trends.sum(axis=1)
    monthly_sentiment_score.index = monthly_sentiment_score.index.to_timestamp()

    return sample_data[['tweet', 'sentiment']], monthly_trends, monthly_sentiment_score

# Streamlit interface
st.title("Sentiment and Spike Detection on Twitter Dataset")
dataset_link = st.text_input("Dataset Link from Hugging Face", placeholder="paste_dataset_link_here")
user_text = st.text_area("Enter text to analyze sentiment directly:")

if st.button("Analyze"):
    if dataset_link:
        data = load_dataset(dataset_link)
        data  = preprocess_tweets(data)
        if data is not None:
            # Run sentiment and trend analysis
            sentiment_data, monthly_trends, monthly_sentiment_score = analyze_sentiment_trend(data)
            tweet_counts, spikes = detect_spikes(data)  # Spike detection

            if sentiment_data is not None:
                # Sentiment Analysis Results
                st.subheader("Sentiment Analysis Results")
                st.write(sentiment_data)

                # Monthly Trends Visualization
                st.subheader("Monthly Sentiment Trends")
                fig, ax = plt.subplots()
                monthly_trends.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
                ax.set_title("Monthly Sentiment Distribution")
                st.pyplot(fig)

                # Monthly Sentiment Score Line Chart
                st.subheader("Monthly Sentiment Score")
                fig, ax = plt.subplots()
                monthly_sentiment_score.plot(ax=ax, color='purple', linestyle='--', marker='o')
                ax.set_title("Monthly Sentiment Score Over Time")
                st.pyplot(fig)

                # Spike Detection Visualization
                st.subheader("Tweet Activity and Spikes")
                fig, ax = plt.subplots()
                tweet_counts.plot(ax=ax, color='skyblue', label='Daily Tweet Counts')
                spikes.plot(ax=ax, style='o', color='red', label='Spikes')
                ax.set_title("Spike Detection in Tweet Activity")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Failed to analyze sentiment and trends.")
        else:
            st.warning("Failed to load the dataset.")
    elif user_text:
        # Direct text analysis
        single_sentiment = get_groq_sentiment(user_text)
        st.write(f"Sentiment of input text: {single_sentiment}")
