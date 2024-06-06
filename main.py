import time
from apify_client import ApifyClient

import nltk
#Used to plot the graphs
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
nltk.download('averaged_perceptron_tagger')
#Used to plot the word cloud
from wordcloud import WordCloud
import re
import requests
from bs4 import BeautifulSoup
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

#Used for Sentimental Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Used for LDA Modelling
from sklearn.feature_extraction.text import CountVectorizer



# Custom CSS to ensure equal spacing
st.markdown("""
    <style>
    .section {
        margin-bottom: 40px;
    }

    </style>
    """, unsafe_allow_html=True)

review_list=[]
headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Function to get the image of the product
def get_product_image(product_name,asin):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    url = f"https://www.amazon.com.au/{product_name}/dp/{asin}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        #img_tag = soup.find('img', class_='a-dynamic-image')
        # Extract the src attribute value
        #if img_tag and 'src' in img_tag.attrs:
         #   img_src = img_tag['src']
        #    return img_src
        #else:
        #    return None
        # Extract the image source
        img_tag = soup.find('img', class_='a-dynamic-image')
        img_src = img_tag['src'] if img_tag and 'src' in img_tag.attrs else None

        # Extract total review count
        review_count_tag = soup.find('span', {'data-hook': 'total-review-count'})
        review_count = review_count_tag.text.strip() if review_count_tag else None

        # Extract rating out of text
        rating_tag = soup.find('span', {'data-hook': 'rating-out-of-text'})
        rating_text = rating_tag.text.strip() if rating_tag else None

        return {
            'image_src': img_src,
            'review_count': review_count,
            'rating_text': rating_text
        }
    else:
        return None

def scrape_amazon_reviews(product_name,asin):
    # Initialize the ApifyClient with your API token (# this needs to changed once trial run is completed)
    client = ApifyClient("apify_api_NQMAGGCjc4vsveJHkpKEpZFSWZuuxF0A4wIb")
    #apify_api_tr5652RiCP7u0jVePrdD8fLUzyWdgH4ywINj
    # Prepare the Actor input
    run_input = {
        "productUrls": [{"url": f"https://www.amazon.com.au/{product_name}/dp/{asin}"}],
        #"maxReviews": 10,
        "sort": "helpful",
        "includeGdprSensitive": False,
        "filterByRatings": ["allStars"],
        "reviewsUseProductVariantFilter": False,
        "reviewsEnqueueProductVariants": False,
        "proxyConfiguration": {"useApifyProxy": True},
        "scrapeProductDetails": False,
        "reviewsAlwaysSaveCategoryData": False,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("R8WeJwLuzLZ6g4Bkk").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    dataset_items = client.dataset(run['defaultDatasetId']).list_items().items
    return dataset_items

def scrape_amazon(search_query, keyword):
    headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    url = f"https://www.amazon.com.au/s?k={search_query}"
    product_info = {}  # dict to get the product info
    response = requests.get(url, headers=headers)
    pattern = r'\b{}\b'.format(re.escape(brand_key))
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        product_links = soup.select('h2 > a.a-link-normal')

        for product in product_links:
            # Extract product name and ASIN from the product link
            href = product['href']
            url_parts = href.split('/dp/')
            if len(url_parts) > 1:
                product_name = url_parts[0].split('/')[-1]
                asin = url_parts[1].split('/')[0]
                if re.search(pattern, product_name, re.IGNORECASE):
                    product_info.update({asin: product_name})
            #else:
             #   print("Failed to extract ASIN from URL: %s", product)
            #   return None, None
        return product_info

    elif response.status_code == 503:
        print("503 error: Service Unavailable. Retrying after a delay...")
        time.sleep(10)  # Delay for 10 seconds before retrying
        return scrape_amazon(search_query, keyword)
    else:
        print("Failed to fetch search results. Status code:", response.status_code)
        return []

def sentiment_analysis(df):
    # Example DataFrame
    df = df
    # tokenization of the reddit column
    reviews_token = df['reviewDescription'].apply(word_tokenize)

    # Assigning the variable against all the token created above and ultimately forming a list of the words
    token_review = [token for tokens in reviews_token for token in tokens]

    # calculating the count of each token against it
    review_count = Counter(token_review)

    # calculating the top 20 common words
    token_list = review_count.most_common(20)

    # Sepeating a list of all the terms and their respective frequency from the above calculated list of tokens
    token_term = [term for term, _ in token_list]
    token_frequencies = [frequency for _, frequency in token_list]

    # Custom contraction mapping to handle common variations (if needed)
    contraction_mapping = {
        "I've": "I have",
        "ive": "I have",
        "didn't": "did not",
        "cant": "cannot",
        "can't": "cannot",
        "don't": "do not",
        "dont": "do not",
        "it's": "it is",
        "its": "it is",
        "I’ll": "I will",
        # Add more contractions as needed
    }

    def expand_contractions(text, contraction_mapping):
        for contraction, expansion in contraction_mapping.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text
    #Method to preprocess custom stopwords

    def read_custom_stopwords(file_path):
        with open(file_path, 'r') as file:
            stopwords_list = [line.strip() for line in file.readlines()]
        return set(stopwords_list)

    # Step 1: Read custom stopwords from file
    custom_stopwords_file = 'custom_stopwords.txt'  # Specify your custom stopwords file
    custom_stop_words = read_custom_stopwords(custom_stopwords_file)

    # Combine NLTK stopwords with custom stopwords
    default_stop_words = set(nltk.corpus.stopwords.words('english'))
    combined_stop_words = default_stop_words.union(custom_stop_words)

    def preprocess_text(text, contraction_mapping, stop_words):
        # Expand contractions using the custom mapping
        text = expand_contractions(text, contraction_mapping)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove punctuation (excluding commas) and other non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)

        # Remove any remaining commas
        text = re.sub(',', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]

        return ' '.join(tokens)

    # Initialize the stop words, stemmer, and lemmatizer
    #stop_words = set(stopwords.words('english'))

    df['Clean_Review'] = df['reviewDescription'].apply(
        lambda text: preprocess_text(text, contraction_mapping, combined_stop_words))
    df = df.dropna()

    # tokenization of the reddit column
    review_token = df['Clean_Review'].apply(word_tokenize)

    # Assigning the variable against all the token created above and ultimately forming a list of the words
    token_review = [token for tokens in review_token for token in tokens]

    # calculating the count of each token against it
    review_count = Counter(token_review)

    # calculating the top 20 common words
    token_list = review_count.most_common(20)

    # Sepeating a list of all the terms and their respective frequency from the above calculated list of tokens
    token_term = [term for term, _ in token_list]
    token_frequencies = [frequency for _, frequency in token_list]

    # Assigning this variable to consider the Reddit Column as a string
    final_text = ' '.join(df['Clean_Review'])

    # Generate word cloud for the whole dataset
    final_cloud = WordCloud(width=800, height=400, background_color='white').generate(final_text)

    # Assignment a variable to hold the VADER function
    final_ana = SentimentIntensityAnalyzer()

    # Creating a list to later append the compound score & sentiment against it
    final_senti = []

    # In order to calculate the compound score against the reddit column
    for index, row in df.iterrows():
        text = row['Clean_Review']
        senti_score = final_ana.polarity_scores(text)
        score = senti_score['compound']

        # Parameter to calcualte the score
        if score >= 0.05:
            sentiment = 'positive'
        elif score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Appending the sentiment to the list
        final_senti.append(sentiment)

    # Finally creating a new column in the dataset to hold the sentiment against each value
    df['sentiment'] = final_senti
    sentiment_count = df['sentiment'].value_counts()

    df['sentiment_score'] = df['Clean_Review'].apply(lambda x: final_ana.polarity_scores(x)['compound'])
    positive_reviews_df = df[df['sentiment_score'] > 0]

    # Step 2: N-gram Extraction

    # Combine all cleaned reviews into a single text
    all_clean_reviews = ' '.join(df['Clean_Review'])

    # Extract bigrams (2-grams) and trigrams (3-grams)
    def extract_ngrams(texts, n=2):
        vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        bigrams_matrix = vectorizer.fit_transform(df['Clean_Review'])
        bigrams = vectorizer.get_feature_names_out()
        bigram_counts = bigrams_matrix.toarray().sum(axis=0)
        bigram_counts_dict = dict(zip(bigrams, bigram_counts))
        return bigram_counts_dict

    # Get the top 10 n-grams
    all_clean_reviews = df['Clean_Review'].tolist()
    bigrams = extract_ngrams(all_clean_reviews, n=2)
    #combined_ngrams = Counter(bigrams)
    #top_ngrams = combined_ngrams.most_common(10)

    # Filter Bigrams using POS Tagging
    def filter_meaningful_ngrams(ngram_counts):
        meaningful_ngrams = []
        for ngram, count in ngram_counts.items():
            words = ngram.split()
            tagged_words = pos_tag(words)
            if ((tagged_words[0][1].startswith('JJ') and tagged_words[1][1].startswith('NN')) or
                    (tagged_words[0][1].startswith('NN') and tagged_words[1][1].startswith('NN'))):
                meaningful_ngrams.append((ngram, count))
        return meaningful_ngrams

    meaningful_top_ngrams = filter_meaningful_ngrams(bigrams)
    meaningful_top_ngrams = sorted(meaningful_top_ngrams, key=lambda x: x[1], reverse=True)[:10]

    # Print Top N-grams as Highlights
    print("Highlights for the Product:")
    for ngram, count in meaningful_top_ngrams:
        print(f"{ngram.capitalize()} ({count} mentions)")

    ####Old code####
    # Step 2: POS Tagging and Keyword Extraction

    stop_words = set(stopwords.words('english'))

    def extract_keywords(review):
        words = word_tokenize(review)
        tagged_words = pos_tag(words)
        keywords = [word for word, tag in tagged_words if tag.startswith('JJ') or tag.startswith('NN')]
        filtered_keywords = [word.lower() for word in keywords if word.lower() not in stop_words]
        return filtered_keywords

    df['keywords'] = df['Clean_Review'].apply(extract_keywords)

    # Step 3: Filter and Rank Keywords
    all_keywords = [keyword for keywords_list in df['keywords'] for keyword in keywords_list]
    filtered_keywords = [keyword for keyword in all_keywords if len(keyword) > 2]  # Filter short words
    keyword_counts = Counter(filtered_keywords)
    top_keywords = keyword_counts.most_common(10)  # Adjust the number based on your preference

    # Print Top Keywords as Highlights
    print("Highlights for the Product:")
    for keyword, count in top_keywords:
        print(f"{keyword.capitalize()} ({count} mentions)")

    # Use columns to split the page into two columns
    col1, col2 = st.columns(2)

    # Section 1: DataFrame (left column)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Product Highlights")
        highlights_text = "\n".join(
            [f"- **{keyword.capitalize()}** ({count} mentions)" for keyword, count in top_keywords])
        st.markdown(highlights_text)

        st.markdown('</div>', unsafe_allow_html=True)
    # Section 2: Line Chart (right column)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(16, 10))
        labels = sentiment_count.keys()
        sizes = sentiment_count.values
        colors = ['green', 'blue', 'red']

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
                                          textprops=dict(fontsize=14, fontweight='bold'))

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # Make the labels and percentages bold
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Bar Chart (left column)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Wordcloud ")
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(final_cloud, interpolation='bilinear')
        ax.set_title("Normal Word Cloud for Final DataFrame")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Scatter Plot (right column)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)

        # st.bar_chart(df)
        st.markdown('</div>', unsafe_allow_html=True)
      #  st.header("Keyword Extraction")

        # Subsection: Positive Keywords
       # st.markdown('<div class="section">', unsafe_allow_html=True)
       # st.subheader("Positive Keywords")
        # positive_keywords = ["good", "excellent", "positive", "happy", "joyful"]
       # st.write("Positive keywords extracted:")
        # st.write(", ".join(positive_keywords))

        # Subsection: Negative Keywords
       # st.subheader("Negative Keywords")
        # negative_keywords = ["bad", "poor", "negative", "sad", "angry"]
       # st.write("Negative keywords extracted:")
        # st.write(", ".join(negative_keywords))
       # st.markdown('</div>', unsafe_allow_html=True)


# Sidebar for user input

if __name__ == "__main__":
   st.title('Curu App Review Analysis')
   st.sidebar.title('Search')
   search_product = st.sidebar.text_input('Enter the brand+product(Eg :"Dove Shampoo")').strip()

   if search_product:
       # Extract brand key from the provided input
       brand_key = search_product.split()[0].lower()

       # Scrape Amazon for product information based on the provided input
       product_info = scrape_amazon(search_product, brand_key)
       print(product_info)
       # Display the product information as a dropdown
       if product_info:
           st.sidebar.title("Product Details")
           selected_product = st.sidebar.selectbox("Select",list(product_info.values()))

           # Get reviews for the selected product
           review_asin = None
           for asin, product_name in product_info.items():
               if product_name == selected_product:
                   review_asin = asin
                   #print(review_asin,product_name)
                   break

           # Scrape reviews for the selected ASIN
           if review_asin:
               reviews = scrape_amazon_reviews(product_name,review_asin)
               # Display the selected product information
               product_details = get_product_image(product_name, review_asin)
               if product_details:
                   # Get product details
                    image = product_details['image_src']
                    review_count = product_details['review_count']
                    rating_text = product_details['rating_text']
               link = "Go to Site"
               url = f"https://www.amazon.com.au/{product_name}/dp/{asin}"
               if image:
                   st.sidebar.image(image, use_column_width=False)
                   st.sidebar.markdown(f"[{link}]({url})")
               else:
                   st.error("Image could not be retrieved.")

               st.markdown(f"**Product Information: {selected_product}**")

               if not reviews:
                   st.write("No reviews available for this product")
               else:
                   review_descriptions = [item['reviewDescription'] for item in reviews]

                   # Create DataFrame
                   df = pd.DataFrame({'reviewDescription': review_descriptions})
                   st.markdown(f"Below insights are from the **{len(df)} reviews** available in **Amazon.**")
                   st.markdown(f"**Overall Product Rating:** {rating_text} (based on total **{review_count}).**")

                   sentiment_analysis(df)
           else:
               st.write("ASIN not found for the selected product.")
       else:
           st.write("No product information found.")


