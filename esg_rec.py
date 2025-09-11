import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_stocks = 100  # Number of stocks
data = pd.DataFrame({
    'ticker': [f'STOCK_{i}' for i in range(n_stocks)],
    'esg_score': np.random.uniform(2, 10, n_stocks).round(1),  # ESG scores: 2-10
    'volatility': np.random.uniform(0.1, 0.5, n_stocks).round(2),  # Volatility: 0.1-0.5
    'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'], n_stocks),
    'price': np.random.uniform(20, 100, n_stocks).round(2)  # Prices: $20-$100
})

def recommend_stocks(esg_min, vol_max, sector=None):
    """Recommend stocks based on ESG score, volatility, and optional sector."""
    filtered = data[(data['esg_score'] >= esg_min) & (data['volatility'] <= vol_max)]
    if sector and sector != "None":
        filtered = filtered[filtered['sector'] == sector]
    if filtered.empty:
        return pd.DataFrame()  # Return empty DataFrame if no matches
    return filtered.sort_values(by='esg_score', ascending=False).head(10)

# Streamlit app
st.title("ESG-Driven Stock Recommender")
st.markdown("Select your preferences to get stock recommendations tailored to ESG and risk.")

# User inputs
esg_min = st.slider("Minimum ESG Score (0-10)", 2.0, 10.0, 5.0, step=0.5)
vol_max = st.slider("Maximum Volatility (Risk)", 0.1, 0.5, 0.3, step=0.05)
sector = st.selectbox("Preferred Sector (Optional)", options=["None", "Technology", "Healthcare", "Finance", "Energy"])

# Get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_stocks(esg_min, vol_max, sector)
    if not recommendations.empty:
        st.subheader("Top Recommended Stocks")
        st.dataframe(recommendations.style.format({"esg_score": "{:.1f}", "volatility": "{:.2f}", "price": "${:.2f}"}))
        # Optional: Save recommendations
        recommendations.to_csv('recommended_stocks.csv', index=False)
        st.download_button("Download Recommendations", data=recommendations.to_csv(index=False), file_name="recommended_stocks.csv")
    else:
        st.warning("No stocks match your criteria. Try relaxing preferences (e.g., lower ESG score or higher volatility).")

# Optional: Visualization
st.subheader("ESG Score Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
data['esg_score'].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("ESG Score")
ax.set_ylabel("Number of Stocks")
st.pyplot(fig)

# Save data for reference
data.to_csv('synthetic_esg_stocks.csv', index=False)