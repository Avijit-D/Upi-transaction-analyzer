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
from fuzzywuzzy import fuzz, process
import unicodedata

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing punctuation and special characters,
    and keeping only alphabetic characters and spaces.
    """
    if pd.isna(text) or text == 'N/A':
        return text
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove punctuation and special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces and normalize unicode
    text = ' '.join(text.split())
    text = unicodedata.normalize('NFKD', text)
    
    return text

def identify_categorical_columns(df):
    """
    Identify columns that contain categorical names/entities.
    """
    categorical_cols = []
    
    for col in df.columns:
        # Check if column contains text data
        if df[col].dtype == 'object':
            # Sample some values to check if they look like names/entities
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                # Check if most values contain alphabetic characters
                alpha_ratio = sum(1 for val in sample_values if re.search(r'[a-zA-Z]', str(val))) / len(sample_values)
                if alpha_ratio > 0.7:  # If more than 70% contain letters
                    categorical_cols.append(col)
    
    return categorical_cols

def fuzzy_group_similar_values(values, similarity_threshold=85):
    """
    Group similar values using fuzzy matching.
    
    Args:
        values: List of unique values to group
        similarity_threshold: Minimum similarity score (0-100) to consider values as similar
    
    Returns:
        tuple: (value_mapping, similarity_examples)
    """
    if len(values) <= 1:
        return {val: val for val in values}, []
    
    # Normalize all values
    normalized_values = {val: normalize_text(val) for val in values}
    
    # Create mapping from original values to representative names
    value_mapping = {}
    processed_values = set()
    similarity_examples = []
    
    for original_val in values:
        if original_val in processed_values:
            continue
            
        normalized_val = normalized_values[original_val]
        similar_values = [original_val]
        
        # Find similar values
        for other_val in values:
            if other_val != original_val and other_val not in processed_values:
                other_normalized = normalized_values[other_val]
                
                # Use different fuzzy matching strategies
                ratio = fuzz.ratio(normalized_val, other_normalized)
                partial_ratio = fuzz.partial_ratio(normalized_val, other_normalized)
                token_sort_ratio = fuzz.token_sort_ratio(normalized_val, other_normalized)
                token_set_ratio = fuzz.token_set_ratio(normalized_val, other_normalized)
                
                # Take the maximum similarity score
                max_similarity = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
                
                if max_similarity >= similarity_threshold:
                    similar_values.append(other_val)
                    # Store example with similarity score
                    if len(similarity_examples) < 5:  # Limit examples
                        similarity_examples.append({
                            'value1': original_val,
                            'value2': other_val,
                            'similarity': max_similarity,
                            'algorithm': 'max'
                        })
        
        # Choose representative name (first encountered or most common)
        representative = similar_values[0]
        
        # Map all similar values to the representative
        for val in similar_values:
            value_mapping[val] = representative
            processed_values.add(val)
    
    return value_mapping, similarity_examples

def apply_fuzzy_normalization(df, similarity_threshold=85):
    """
    Apply fuzzy normalization to categorical columns in the dataset.
    
    Args:
        df: DataFrame to normalize
        similarity_threshold: Minimum similarity score for grouping (0-100)
    
    Returns:
        tuple: (normalized DataFrame, normalization details)
    """
    df = df.copy()
    normalization_details = {}
    
    # Identify categorical columns
    categorical_cols = identify_categorical_columns(df)
    
    if not categorical_cols:
        return df, normalization_details
    
    # Apply normalization to each categorical column
    for col in categorical_cols:
        if col in df.columns:
            # Get unique values
            unique_values = df[col].dropna().unique()
            
            if len(unique_values) > 1:
                # Create mapping for this column
                value_mapping, similarity_examples = fuzzy_group_similar_values(unique_values, similarity_threshold)
                
                # Apply mapping
                df[col] = df[col].map(value_mapping).fillna(df[col])
                
                # Store normalization details
                normalization_details[col] = {
                    'original_count': len(unique_values),
                    'normalized_count': len(df[col].dropna().unique()),
                    'reduction': len(unique_values) - len(df[col].dropna().unique()),
                    'groupings': value_mapping,
                    'similarity_examples': similarity_examples
                }
    
    return df, normalization_details

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

# Add a placeholder for normalization stats
normalization_stats_placeholder = st.empty()

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
    
    # Apply fuzzy normalization to categorical fields
    with st.spinner("Processing and normalizing data..."):
        df, normalization_details = apply_fuzzy_normalization(df, 85)  # Default threshold
    
    # Update normalization statistics
    if normalization_details:
        total_original = sum(details['original_count'] for details in normalization_details.values())
        total_normalized = sum(details['normalized_count'] for details in normalization_details.values())
        total_reduction = total_original - total_normalized
        
        if total_reduction > 0:
            normalization_stats_placeholder.success(
                f"**Normalization Results:**\n"
                f"Reduced from {total_original} to {total_normalized} unique values\n"
                f"({total_reduction} values grouped)"
            )
        else:
            normalization_stats_placeholder.info("No values were grouped during normalization")
    else:
        normalization_stats_placeholder.info("No categorical columns processed")
    
    # Display normalization results in an expander
    with st.expander("View Normalization Details", expanded=False):
        st.write("**Normalization Summary:**")
        
        if normalization_details:
            for col, details in normalization_details.items():
                if details['reduction'] > 0:
                    st.write(f"- **{col}**: Reduced from {details['original_count']} to {details['normalized_count']} unique values ({details['reduction']} values grouped)")
                    
                    # Show some example groupings
                    if details['groupings']:
                        st.write(f"  **Example groupings in {col}:**")
                        # Show first few groupings as examples
                        example_count = 0
                        for original, representative in details['groupings'].items():
                            if original != representative and example_count < 3:
                                st.write(f"    '{original}' → '{representative}'")
                                example_count += 1
                        if example_count == 0:
                            st.write("    No values were grouped together.")
                        
                        # Show similarity examples with scores
                        if 'similarity_examples' in details and details['similarity_examples']:
                            st.write(f"  **Similarity examples (threshold: 85%):**")
                            for example in details['similarity_examples'][:3]:  # Show first 3
                                st.write(f"    '{example['value1']}' ↔ '{example['value2']}' (similarity: {example['similarity']:.1f}%)")
                else:
                    st.write(f"- **{col}**: No similar values found to group")
        else:
            st.write("No categorical columns were processed for normalization.")
        
        st.write("**Note:** Similar names, merchants, and payees have been grouped together to improve analysis accuracy.")
        st.write("Examples of what gets grouped:")
        st.write("- 'Amazon Pay', 'amazonpay', 'AMZN Pay' → 'Amazon Pay'")
        st.write("- 'John Doe', 'john doe', 'JOHN DOE' → 'John Doe'")
        st.write("- 'Uber', 'UBER', 'uber' → 'Uber'")
    
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
    
    # Fuzzy normalization settings
    st.subheader("Fuzzy Normalization Settings")
    fuzzy_similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=70,
        max_value=95,
        value=85,
        step=5,
        help="Minimum similarity score to group similar names/entities. Higher values = stricter grouping."
    )
    
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
    
