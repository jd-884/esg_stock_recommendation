import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import time

# Streamlit page config
st.set_page_config(page_title="ESG Stock Recommender", layout="wide")

# Fetch real stock data from yfinance
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CSCO',
    'INTC', 'AMD', 'ORCL', 'CRM', 'IBM', 'QCOM', 'TXN', 'AVGO', 'INTU', 'NOW',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'TROW',
    'SPGI', 'MCO', 'PNC', 'USB', 'COF', 'UNH', 'PFE', 'MRK', 'JNJ', 'ABT',
    'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'LLY', 'ABBV', 'CVS', 'CI', 'MDT',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'VLO', 'PSX', 'MPC', 'HES',
    'PG', 'KO', 'PEP', 'WMT', 'TGT', 'COST', 'MCD', 'SBUX', 'CL', 'KMB',
    'NKE', 'HD', 'LOW', 'TJX', 'BKNG', 'CMG', 'DRI', 'YUM', 'LULU', 'EBAY',
    'BA', 'LMT', 'CAT', 'DE', 'GE', 'HON', 'UPS', 'RTX', 'MMM', 'CSX',
    'DOW', 'DD', 'NEM', 'FCX', 'APD', 'NEE', 'DUK', 'SO', 'D', 'AEP',
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA'
]

# Show loading indicator
with st.spinner("Fetching stock data..."):
    data = []
    for ticker in tickers:
        for _ in range(2):  # Retry twice
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1y')
                if not hist.empty:
                    data.append({
                        'ticker': ticker,
                        'price': hist['Close'].iloc[-1].round(2),
                        'volatility': (hist['Close'].pct_change().std() * np.sqrt(252)).round(2),
                        'sector': stock.info.get('sector', 'Unknown'),
                        'esg_score': np.random.uniform(2, 10, 1)[0].round(1)  # Synthetic ESG
                    })
                    break
            except Exception:
                time.sleep(1)
                continue
    if not data:
        st.warning("Failed to fetch yfinance data. Using synthetic data.")
        np.random.seed(42)
        n_stocks = 100
        data = pd.DataFrame({
            'ticker': [f'STOCK_{i}' for i in range(n_stocks)],
            'esg_score': np.random.uniform(2, 10, n_stocks).round(1),
            'volatility': np.random.uniform(0.1, 0.5, n_stocks).round(2),
            'sector': np.random.choice(['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer', 'Industrials', 'Materials', 'Utilities', 'Real Estate'], n_stocks),
            'price': np.random.uniform(20, 200, n_stocks).round(2)
        })
    else:
        data = pd.DataFrame(data)

# Clustering: Group stocks by ESG and volatility
# 1. Select features (ESG score, volatility)
X = data[['esg_score', 'volatility']]
# 2. Standardize features to same scale (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 3. Apply K-Means to create 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)  # Assign cluster labels (0, 1, 2)

def recommend_stocks(esg_min, vol_max, sector=None):
    """Recommend stocks based on ESG score, volatility, and sector."""
    # Map user inputs to closest cluster
    user_input = pd.DataFrame([[esg_min, vol_max]], columns=['esg_score', 'volatility'])
    user_input_scaled = scaler.transform(user_input)
    user_cluster = kmeans.predict(user_input_scaled)[0]
    # Filter stocks in the user's cluster
    filtered = data[data['cluster'] == user_cluster]
    # Apply ESG and volatility filters
    filtered = filtered[(filtered['esg_score'] >= esg_min) & (filtered['volatility'] <= vol_max)]
    if sector and sector != "None":
        filtered = filtered[filtered['sector'] == sector]
    if filtered.empty:
        return pd.DataFrame()
    return filtered.sort_values(by='esg_score', ascending=False).head(10)

# Streamlit app
st.title("ESG-Driven Stock Recommender")
st.markdown("Explore sustainable investments using real-time financial data from Yahoo Finance.")

# User inputs
esg_min = st.slider("Minimum ESG Score (0-10)", 2.0, 10.0, 5.0, step=0.5, help="Higher scores prioritize sustainability.")
vol_max = st.slider("Maximum Volatility (Risk)", 0.1, 2.0, 0.3, step=0.05, help="Lower values prioritize stability.")
sector_options = ["None"] + sorted(data['sector'].unique().tolist())
sector = st.selectbox("Preferred Sector (Optional)", options=sector_options, help="Filter by industry sector.")

# Cluster summary
st.subheader("Cluster Overview")
cluster_summary = data.groupby('cluster')[['esg_score', 'volatility']].mean().round(2)
cluster_labels = {0: "High ESG, Low Risk", 1: "Moderate ESG, Moderate Risk", 2: "Low ESG, High Risk"}
cluster_summary['Description'] = [cluster_labels.get(i, f"Cluster {i}") for i in cluster_summary.index]
st.dataframe(cluster_summary.style.format({"esg_score": "{:.1f}", "volatility": "{:.2f}"}))

# Get recommendations
if st.button("Get Recommendations", help="Generate stock picks"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_stocks(esg_min, vol_max, sector)
        if not recommendations.empty:
            st.subheader("Top Recommended Stocks")
            cluster_num = recommendations['cluster'].iloc[0]
            st.write(f"Selected Cluster: {cluster_labels.get(cluster_num, f'Cluster {cluster_num}')} (Avg ESG: {cluster_summary.loc[cluster_num, 'esg_score']:.1f}, Avg Volatility: {cluster_summary.loc[cluster_num, 'volatility']:.2f})")
            st.dataframe(recommendations.style.format({"esg_score": "{:.1f}", "volatility": "{:.2f}", "price": "${:.2f}"}))
            # Save and download
            recommendations.to_csv('recommended_stocks.csv', index=False)
            st.download_button(
                label="Download Recommendations",
                data=recommendations.to_csv(index=False),
                file_name="recommended_stocks.csv",
                mime="text/csv",
                help="Download as CSV"
            )
            # Scatter plot
            st.subheader("Recommended Stocks: ESG vs. Volatility")
            fig, ax = plt.subplots(figsize=(8, 4))
            for sector in recommendations['sector'].unique():
                subset = recommendations[recommendations['sector'] == sector]
                ax.scatter(subset['volatility'], subset['esg_score'], label=sector, s=100)
                for i, row in subset.head(5).iterrows():  # Limit labels to avoid clutter
                    ax.text(row['volatility'], row['esg_score'], row['ticker'], fontsize=8)
            ax.set_xlabel("Volatility (Risk)")
            ax.set_ylabel("ESG Score")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No stocks match your criteria. Try relaxing preferences (e.g., lower ESG score or higher volatility).")

# ESG score distribution
st.subheader("ESG Score Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
data['esg_score'].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("ESG Score")
ax.set_ylabel("Number of Stocks")
st.pyplot(fig)

# Save data
data.to_csv('esg_stocks_yfinance.csv', index=False)