import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
df = pd.read_csv('session_data.csv')

# Load Apriori model and label encoder
with open('apriori_model.pkl', 'rb') as file:
    rules, label_encoder = pkl.load(file)

# Function to recommend products based on association rules


def get_ar_recommendations(cart_products, rules, label_encoder, num_recommendations=3):
    cart_product_ids = label_encoder.transform(cart_products)
    recommendations = pd.DataFrame()
    for product_idx in cart_product_ids:
        product_rules = rules[rules['antecedents'].apply(
            lambda x: product_idx in x)]
        product_rules = product_rules.sort_values(
            'confidence', ascending=False)
        recommendations = recommendations.append(product_rules)
    recommendations = recommendations.drop_duplicates(subset='consequents')
    recommendations = recommendations.sort_values(
        'confidence', ascending=False).head(num_recommendations)
    recommended_products = []
    for _, row in recommendations.iterrows():
        consequent_products = row['consequents']
        for product in consequent_products:
            if product not in cart_product_ids:
                recommended_products.append(
                    label_encoder.inverse_transform([product])[0])
    return recommended_products


# Streamlit app
st.set_page_config(page_title="Product Recommendation System",
                   page_icon="🛒", layout="wide")

st.title('🛒 Product Recommendation System')
st.write("Get product recommendations based on the items in your cart.")

st.sidebar.header("Input")
cart_products = st.sidebar.text_input(
    'Enter items in your cart (comma-separated)').split(',')

cart_products = [product.strip()
                 for product in cart_products if product.strip()]

st.sidebar.header("Settings")
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 3)

if st.sidebar.button("Predict"):
    if cart_products:
        predictions = get_ar_recommendations(
            cart_products, rules, label_encoder, num_recommendations=top_n)

        st.subheader("Recommended Additions: Powered by Apriori")
        for product in predictions:
            st.write(f"🔹 {product}")
    else:
        st.write("⚠️ Please enter at least one product.")

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: black;
            border-top: 1px solid #ddd;
        }
    </style>
    <div class="footer">
        <p>Developed by Anirvan Krishna (21EE38002) for Zepto Summer Internships</p>
    </div>
""", unsafe_allow_html=True)
