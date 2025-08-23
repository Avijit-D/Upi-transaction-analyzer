import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def parse_html_to_dataframe(file_content):
    soup = BeautifulSoup(file_content, 'html.parser')
    transactions = []

    for transaction in soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1'):
        transaction_text = transaction.get_text(separator=" ").strip()

        paid_pattern = r'Paid ₹([\d,]+\.\d{2}) to (.+?) using Bank Account (XXXXXX\d{4})'
        sent_pattern = r'Sent ₹([\d,]+\.\d{2}) using Bank Account (XXXXXX\d{4})'
        received_pattern = r'Received ₹([\d,]+\.\d{2})'
        date_time_pattern = r'(\d{1,2} \w+ \d{4}, \d{1,2}:\d{2}:\d{2} \w{3})'
        date_time_match = re.search(date_time_pattern, transaction_text)

        if date_time_match:
            date_time = date_time_match.group(1)
            transaction_text = transaction_text[:date_time_match.start()].strip()
            date_str, time_str = date_time.split(', ')
            time_str = time_str.strip()
        else:
            continue

        match = re.search(paid_pattern, transaction_text)
        if match:
            amount = match.group(1)
            payee = match.group(2)
            account = match.group(3)
            transactions.append(['Paid', payee, amount, account, date_str, time_str])
            continue

        match = re.search(sent_pattern, transaction_text)
        if match:
            amount = match.group(1)
            account = match.group(2)
            transactions.append(['Sent', 'N/A', amount, account, date_str, time_str])
            continue

        match = re.search(received_pattern, transaction_text)
        if match:
            amount = match.group(1)
            transactions.append(['Received', 'N/A', amount, 'N/A', date_str, time_str])
            continue

    df = pd.DataFrame(transactions, columns=['Transaction Type', 'Payee', 'Amount', 'Account', 'Date', 'Time'])
    df['Amount'] = df['Amount'].str.replace(',', '').astype(float)
    return df

def optimal_k_means(X, max_k=20):
    best_k = 2
    best_score = -1
    for k in range(2, min(len(X), max_k) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    final_model = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_model.fit_predict(X)
    return final_labels, best_k

def amount_based_clustering_auto(df):
    df = df.copy()
    df['Log Amount'] = np.log1p(df['Amount'])
    X = df['Log Amount'].values.reshape(-1, 1)
    labels, best_k = optimal_k_means(X)
    df['Amount Cluster'] = [f'Group {i+1}' for i in labels]
    return df, best_k

def plot_amount_clusters(df):
    # Violin plot
    fig1 = px.violin(
        df,
        x='Amount Cluster',
        y='Amount',
        box=True,
        points='all',
        color='Amount Cluster',
        title='Transaction Amount Groups',
        labels={'Amount Cluster': 'Group', 'Amount': 'Amount (₹)'}
    )
    fig1.update_layout(width=800, height=500)
    
    # Scatter plot
    fig2 = px.scatter(
        df,
        x='Date',
        y='Amount',
        color='Amount Cluster',
        title='Transaction Distribution Over Time',
        labels={'Amount': 'Amount (₹)', 'Date': 'Date', 'Amount Cluster': 'Group'},
        hover_data=['Transaction Type', 'Payee']
    )
    fig2.update_layout(width=800, height=500)
    fig2.update_traces(marker=dict(size=8))
    
    return fig1, fig2

def extract_temporal_features(df):
    df = df.copy()
    # Extract hour from time
    df['Hour'] = df['Time'].dt.hour
    
    # Extract weekday and month
    df['Weekday'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    
    return df

def encode_transaction_features(df):
    df = df.copy()
    
    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=['Transaction Type'], prefix='Type')
    
    # Frequency encode payee
    payee_freq = df['Payee'].value_counts(normalize=True)
    df['Payee_Frequency'] = df['Payee'].map(payee_freq)
    
    # Group rare payees
    payee_counts = df['Payee'].value_counts()
    rare_payees = payee_counts[payee_counts < 3].index
    df['Payee_Grouped'] = df['Payee'].replace(rare_payees, 'Others')
    
    # Label encode account
    le = LabelEncoder()
    df['Account_Encoded'] = le.fit_transform(df['Account'].fillna('Unknown'))
    
    # Extract keywords from payee
    vectorizer = CountVectorizer(max_features=10, stop_words='english')
    payee_keywords = vectorizer.fit_transform(df['Payee_Grouped'].fillna('Unknown'))
    keywords_df = pd.DataFrame(payee_keywords.toarray(), 
                             columns=[f'Keyword_{i}' for i in range(payee_keywords.shape[1])])
    df = pd.concat([df, keywords_df], axis=1)
    
    return df

def calculate_frequency_features(df):
    df = df.copy()
    
    # Calculate transaction frequency with same payee
    df['Payee_Frequency_Count'] = df.groupby('Payee')['Payee'].transform('count')
    
    # Calculate transaction frequency with same account
    df['Account_Frequency_Count'] = df.groupby('Account')['Account'].transform('count')
    
    return df

def prepare_feature_matrix(df):
    # Select features for clustering
    features = [
        'Hour',
        'Payee_Frequency',
        'Account_Encoded',
        'Payee_Frequency_Count',
        'Account_Frequency_Count'
    ] + [col for col in df.columns if col.startswith('Type_')] + \
       [col for col in df.columns if col.startswith('Keyword_')]
    
    X = df[features].copy()
    
    # Fill NaN values with appropriate defaults
    X = X.fillna({
        'Hour': X['Hour'].median(),
        'Payee_Frequency': 0,
        'Account_Encoded': 0,
        'Payee_Frequency_Count': 1,
        'Account_Frequency_Count': 1
    })
    
    # Fill NaN values in keyword columns with 0
    keyword_cols = [col for col in X.columns if col.startswith('Keyword_')]
    X[keyword_cols] = X[keyword_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def optimal_k_means_behavioral(X, max_k=20):
    best_k = 2
    best_score = -1
    for k in range(2, min(len(X), max_k) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    final_model = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_model.fit_predict(X)
    return final_labels, best_k

def plot_behavioral_clusters(df, X):
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Behavior Group': df['Behavior Group'],
        'Amount': df['Amount'],
        'Payee': df['Payee'],
        'Transaction Type': df['Transaction Type'].str.split('_').str[-1]
    })
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Behavior Group',
        size='Amount',
        hover_data=['Payee', 'Transaction Type', 'Amount'],
        title='Behavioral Clusters (PCA Reduced)',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
    )
    fig.update_layout(width=800, height=600)
    return fig

def behavioral_clustering(df):
    df = df.copy()
    
    # Extract and engineer features
    df = extract_temporal_features(df)
    df = encode_transaction_features(df)
    df = calculate_frequency_features(df)
    
    # Prepare feature matrix
    X = prepare_feature_matrix(df)
    
    # Find optimal clusters
    labels, num_groups = optimal_k_means_behavioral(X)
    df['Behavior Group'] = [f'Group {i+1}' for i in labels]
    
    return df, num_groups, X

# Set page config
st.set_page_config(
    page_title="Google Pay Analysis",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Google Pay Transaction Analysis")
st.write("Upload your Google Pay activity HTML file to analyze your transaction patterns.")

# File uploader
uploaded_file = st.file_uploader("Choose a Google Pay activity HTML file", type=['html'])

if uploaded_file is not None:
    # Read and parse the HTML file
    file_content = uploaded_file.read().decode('utf-8')
    df = parse_html_to_dataframe(file_content)
    
    # Data preprocessing
    df['Time'] = df['Time'].str[:-4]
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Display basic information
    st.subheader("Data Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Total Amount", f"₹{df['Amount'].sum():,.2f}")
    with col3:
        st.metric("Average Amount", f"₹{df['Amount'].mean():,.2f}")
    with col4:
        st.metric("First Date", df['Date'].min().strftime('%Y-%m-%d'))
    with col5:
        st.metric("Last Date", df['Date'].max().strftime('%Y-%m-%d'))
    
    # Calculate quartiles and IQR for outlier detection
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 5.5 * IQR
    
    # Ensure bounds are within actual data range
    lower_bound = max(lower_bound, df['Amount'].min())
    upper_bound = min(upper_bound, df['Amount'].max())
    
    # Amount range selection
    st.subheader("Transaction Amount Range")
    min_amount, max_amount = st.slider(
        "Select Amount Range (₹)",
        min_value=float(df['Amount'].min()),
        max_value=float(df['Amount'].max()),
        value=(float(lower_bound), float(upper_bound)),
        step=1.0
    )
    
    # Filter data based on amount range
    df_filtered = df[(df['Amount'] >= min_amount) & (df['Amount'] <= max_amount)]
    
    # Payee Selection
    st.subheader("Transaction Details")
    
    # Get unique payees and add a dropdown
    unique_payees = sorted(df_filtered['Payee'].unique())
    selected_payee = st.selectbox(
        "Select Payee",
        ["All"] + [payee for payee in unique_payees if payee != 'N/A']
    )
    
    # Filter by selected payee
    if selected_payee != "All":
        df_filtered = df_filtered[df_filtered['Payee'] == selected_payee]
    
    # Display filtered metrics
    st.write(f"Showing transactions between ₹{min_amount:,.2f} and ₹{max_amount:,.2f}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Transactions", len(df_filtered))
    with col2:
        st.metric("Filtered Total Amount", f"₹{df_filtered['Amount'].sum():,.2f}")
    with col3:
        st.metric("Filtered Average Amount", f"₹{df_filtered['Amount'].mean():,.2f}")
    
    # Search functionality
    st.subheader("Search Transactions")
    search_query = st.text_input(
        "Search by Payee, Amount, Date, or Transaction Type",
        placeholder="e.g., 'John', '1000', '2024-01', 'Paid'"
    )
    
    if search_query:
        # Convert search query to lowercase for case-insensitive search
        search_query = search_query.lower()
        
        # Create a copy of the filtered dataframe for searching
        search_df = df_filtered.copy()
        
        # Convert all columns to strings and lowercase for searching
        search_df['Payee'] = search_df['Payee'].astype(str).str.lower()
        search_df['Amount'] = search_df['Amount'].astype(str)
        search_df['Date'] = search_df['Date'].dt.strftime('%Y-%m-%d')
        search_df['Transaction Type'] = search_df['Transaction Type'].str.lower()
        
        # Search across multiple columns
        mask = (
            search_df['Payee'].str.contains(search_query) |
            search_df['Amount'].str.contains(search_query) |
            search_df['Date'].str.contains(search_query) |
            search_df['Transaction Type'].str.contains(search_query)
        )
        
        search_results = df_filtered[mask]
        
        if len(search_results) > 0:
            # Format the results for display
            display_df = search_results.copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"₹{x:,.2f}")
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Time'] = display_df['Time'].dt.strftime('%H:%M:%S')
            display_df = display_df[['Date', 'Time', 'Transaction Type', 'Payee', 'Amount', 'Account']]
            display_df = display_df.sort_values(['Date', 'Time'], ascending=[False, False])
            
            st.write(f"Found {len(search_results)} matching transactions:")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transactions found matching your search criteria.")
    else:
        st.info("Enter a search term to find specific transactions.")
    
    # Amount-based clustering with automatic k determination
    st.subheader("Transaction Amount Groups")
    
    # Apply clustering to filtered data
    df_filtered, num_groups = amount_based_clustering_auto(df_filtered)
    
    # Display number of groups found
    st.write(f"Automatically determined {num_groups} transaction groups based on amount patterns")
    
    # Calculate and display group statistics
    group_stats = df_filtered.groupby('Amount Cluster')['Amount'].agg(['min', 'max', 'mean', 'count']).round(2)
    group_stats.columns = ['Minimum Amount', 'Maximum Amount', 'Average Amount', 'Transaction Count']
    
    st.write("Group Statistics:")
    st.dataframe(group_stats.style.format({
        'Minimum Amount': '₹{:.2f}',
        'Maximum Amount': '₹{:.2f}',
        'Average Amount': '₹{:.2f}'
    }))
    
    # Interactive cluster visualizations
    violin_fig, scatter_fig = plot_amount_clusters(df_filtered)
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Distribution Analysis", "Time Series Analysis"])
    
    with tab1:
        st.plotly_chart(violin_fig, use_container_width=True)
    
    with tab2:
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Transaction Value Trends
    st.subheader("Transaction Value Trends")
    time_period = st.selectbox(
        "Select Time Period",
        ["Daily", "Weekly", "Monthly"]
    )
    
    if time_period == "Daily":
        daily_amounts = df_filtered.groupby('Day').agg(
            Total_Amount=('Amount', 'sum'),
            Average_Amount=('Amount', 'mean')
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(daily_amounts['Day'], daily_amounts['Total_Amount'], color='skyblue')
        ax1.set_xlabel('Day')
        ax1.set_xticks(range(1, 32))
        ax1.set_ylabel('Total Amount (₹)')
        ax1.set_title('Daily Transaction Amounts')
        
        ax2 = ax1.twinx()
        ax2.plot(daily_amounts['Day'], daily_amounts['Average_Amount'], color='red')
        ax2.set_ylabel('Average Amount per Transaction (₹)')
        st.pyplot(fig)
    
    elif time_period == "Weekly":
        weekly_amounts = df_filtered.groupby('Week').agg(
            Total_Amount=('Amount', 'sum'),
            Average_Amount=('Amount', 'mean')
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(weekly_amounts['Week'], weekly_amounts['Total_Amount'], color='skyblue')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Total Amount (₹)')
        ax1.set_title('Weekly Transaction Amounts')
        
        ax2 = ax1.twinx()
        ax2.plot(weekly_amounts['Week'], weekly_amounts['Average_Amount'], color='red')
        ax2.set_ylabel('Average Amount per Transaction (₹)')
        st.pyplot(fig)
    
    elif time_period == "Monthly":
        monthly_amounts = df_filtered.groupby('Month').agg(
            Total_Amount=('Amount', 'sum'),
            Average_Amount=('Amount', 'mean')
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(monthly_amounts['Month'], monthly_amounts['Total_Amount'], color='skyblue')
        ax1.set_xlabel('Month')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.set_ylabel('Total Amount (₹)')
        ax1.set_title('Monthly Transaction Amounts')
        
        ax2 = ax1.twinx()
        ax2.plot(monthly_amounts['Month'], monthly_amounts['Average_Amount'], color='red')
        ax2.set_ylabel('Average Amount per Transaction (₹)')
        st.pyplot(fig)
    
    # Top Transactions Analysis
    st.subheader("Top Transactions Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Most Paid To",
        "Most Received From",
        "Highest Transactions",
        "Most Frequent Transactions"
    ])
    
    with tab1:
        st.write("Top 10 Payees by Total Amount Paid")
        paid_df = df_filtered[df_filtered['Transaction Type'].isin(['Paid', 'Sent'])]
        top_paid = paid_df.groupby('Payee').agg({
            'Amount': ['sum', 'count'],
            'Transaction Type': 'first'
        }).round(2)
        top_paid.columns = ['Total Amount', 'Transaction Count', 'Type']
        top_paid = top_paid.sort_values('Total Amount', ascending=False).head(10)
        st.dataframe(
            top_paid.style.format({
                'Total Amount': '₹{:.2f}'
            }),
            use_container_width=True
        )
    
    with tab2:
        st.write("Top 10 Recipients by Total Amount Received")
        received_df = df_filtered[df_filtered['Transaction Type'] == 'Received']
        top_received = received_df.groupby('Payee').agg({
            'Amount': ['sum', 'count'],
            'Transaction Type': 'first'
        }).round(2)
        top_received.columns = ['Total Amount', 'Transaction Count', 'Type']
        top_received = top_received.sort_values('Total Amount', ascending=False).head(10)
        st.dataframe(
            top_received.style.format({
                'Total Amount': '₹{:.2f}'
            }),
            use_container_width=True
        )
    
    with tab3:
        st.write("Top 10 Highest Value Transactions")
        highest_transactions = df_filtered.sort_values('Amount', ascending=False).head(10)
        display_df = highest_transactions[['Date', 'Time', 'Transaction Type', 'Payee', 'Amount']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Time'] = display_df['Time'].dt.strftime('%H:%M:%S')
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.write("Top 10 Most Frequent Transactions")
        frequent_transactions = df_filtered.groupby(['Payee', 'Transaction Type']).agg({
            'Amount': ['count', 'sum', 'mean']
        }).round(2)
        frequent_transactions.columns = ['Transaction Count', 'Total Amount', 'Average Amount']
        frequent_transactions = frequent_transactions.sort_values('Transaction Count', ascending=False).head(10)
        st.dataframe(
            frequent_transactions.style.format({
                'Total Amount': '₹{:.2f}',
                'Average Amount': '₹{:.2f}'
            }),
            use_container_width=True
        )
    
    # Behavioral Clustering Analysis
    st.subheader("Behavioral Clustering Analysis")
    
    # Check if we have enough data for behavioral clustering
    if len(df_filtered) >= 10:  # Need at least 10 transactions for meaningful clustering
        # Apply behavioral clustering
        df_behavioral, num_behavioral_groups, X_behavioral = behavioral_clustering(df_filtered)
        
        st.write(f"Automatically determined {num_behavioral_groups} behavioral groups based on transaction patterns")
        
        # Display behavioral group statistics
        behavioral_stats = df_behavioral.groupby('Behavior Group').agg({
            'Amount': ['count', 'sum', 'mean'],
            'Payee': 'nunique',
            'Transaction Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'
        }).round(2)
        behavioral_stats.columns = ['Transaction Count', 'Total Amount', 'Average Amount', 'Unique Payees', 'Most Common Type']
        
        st.write("Behavioral Group Statistics:")
        st.dataframe(behavioral_stats.style.format({
            'Total Amount': '₹{:.2f}',
            'Average Amount': '₹{:.2f}'
        }))
        
        # Create behavioral clustering visualization
        behavioral_fig = plot_behavioral_clusters(df_behavioral, X_behavioral)
        st.plotly_chart(behavioral_fig, use_container_width=True)
        
        # Show sample transactions from each behavioral group
        st.write("Sample Transactions by Behavioral Group:")
        for group in sorted(df_behavioral['Behavior Group'].unique()):
            group_transactions = df_behavioral[df_behavioral['Behavior Group'] == group].head(5)
            st.write(f"**{group}:**")
            display_df = group_transactions[['Date', 'Time', 'Transaction Type', 'Payee', 'Amount']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Time'] = display_df['Time'].dt.strftime('%H:%M:%S')
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.write("---")
    else:
        st.info("Need at least 10 transactions for behavioral clustering analysis.")
    
    # Visualization selection
    st.subheader("Number of Transactions")
    visualization_type = st.selectbox(
        "Select Visualization Type",
        ["By Hour", "By Day", "By Month", "Amount Trends"]
    )
    
    if visualization_type == "By Hour":
        st.subheader("Transactions by Hour")
        hourly_transactions = df_filtered.groupby(['Hour', 'Transaction Type']).size().reset_index(name='Count')
        pivot_df = hourly_transactions.pivot(index="Hour", columns="Transaction Type", values="Count").fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(pivot_df.index, pivot_df["Paid"], label="Paid", color="blue")
        ax.bar(pivot_df.index, pivot_df["Sent"], bottom=pivot_df["Paid"], label="Sent", color="green")
        ax.bar(pivot_df.index, pivot_df["Received"], bottom=pivot_df["Paid"] + pivot_df["Sent"], label="Received", color="purple")
        ax.set_title('Transactions by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Transactions')
        ax.set_xticks(range(24))
        ax.legend(title='Transaction Type')
        st.pyplot(fig)
    
    elif visualization_type == "By Day":
        st.subheader("Transactions by Day")
        day_transactions = df_filtered.groupby('Day').agg(
            Count=('Payee', 'size'),
            Amount=('Amount', 'sum')
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(day_transactions['Day'], day_transactions['Count'], color='skyblue')
        ax1.set_xlabel('Day')
        ax1.set_xticks(range(1, 32))
        ax1.set_ylabel('Number of Transactions')
        ax1.set_title('Transactions by Day')
        
        ax2 = ax1.twinx()
        ax2.plot(day_transactions['Day'], day_transactions['Amount']/day_transactions['Count'], color='red')
        ax2.set_ylabel('Average Amount per Transaction')
        st.pyplot(fig)
    
    elif visualization_type == "By Month":
        st.subheader("Transactions by Month")
        monthly_transactions = df_filtered.groupby('Month').size().reset_index(name='Count')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(monthly_transactions['Month'], monthly_transactions['Count'], color='pink')
        ax.set_xlabel('Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_ylabel('Number of Transactions')
        ax.set_title('Transactions by Month')
        st.pyplot(fig)
    
    elif visualization_type == "Amount Trends":
        st.subheader("Transaction Amount Trends")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_filtered['Date'], df_filtered['Amount'])
        ax.set_title('Transaction Amounts Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount (₹)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df_filtered.describe())
    