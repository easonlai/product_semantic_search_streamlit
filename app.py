# Import necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
openai.api_type = "azure"
openai.api_base = "https://PLESAE_ENTER_YOUR_OWNED_AOAI_RESOURCE_NAME.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
engine = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"

# Reload the item CSV file with the word embedding result.
df = pd.read_csv('./data/grocery_store_items_word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array) # type: ignore

# Set web page header and title.
st.set_page_config(
    page_title="Product Semantic Search"
)
st.title("Product Semantic Search")

#  Create a text input box for the user to enter the search terms.
user_input = st.text_input("Please let me know what you are looking for: ")

# Set the number of cards per row.
N_cards_per_row = 3

# If the user has entered the search terms, then perform the semantic search.
if user_input:
    # Get the word embedding of the user input.
    search_terms_vector = get_embedding(user_input, engine=engine)
    # Calculate the cosine similarity between each item name and the user input.
    df["similarity"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_terms_vector))
    # Display the top 6 semantic-related items by highest similarity.
    df_result = df.sort_values("similarity", ascending=False).head(6)
    for n_row, row in df_result.reset_index().iterrows():
        i = n_row%N_cards_per_row # type: ignore
        if i==0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # draw the card
        with cols[n_row%N_cards_per_row]: # type: ignore
            st.caption(f"**{row['Item'].strip()}**")
            st.markdown(f"*{row['similarity']}*")
            st.image(f"{row['image']}", width = 200)
            st.button("Add to cart", key=cols[n_row%N_cards_per_row]) # type: ignore