import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

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

def amount_based_clustering(df, k=3):
    df = df.copy()
    df['Log Amount'] = np.log1p(df['Amount'])
    X = df['Log Amount'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Amount Cluster'] = kmeans.fit_predict(X)
    return df

def plot_amount_clusters(df):
    fig = px.violin(
        df,
        x='Amount Cluster',
        y='Amount',
        box=True,
        points='all',
        color='Amount Cluster',
        title='Transaction Amount Distribution per Cluster',
        labels={'Amount Cluster': 'Cluster', 'Amount': 'Amount (₹)'}
    )
    fig.update_layout(width=800, height=500)
    return fig

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

    # Amount-based clustering on filtered data
    st.subheader("Transaction Amount Clustering")
    
    # Apply clustering to filtered data
    df_filtered = amount_based_clustering(df_filtered)
    
    # Sort clusters by mean amount
    cluster_means = df_filtered.groupby('Amount Cluster')['Amount'].mean().sort_values()
    cluster_mapping = {old: new for new, old in enumerate(cluster_means.index)}
    df_filtered['Amount Cluster'] = df_filtered['Amount Cluster'].map(cluster_mapping)
    
    # Create cluster labels based on amount ranges
    cluster_stats = df_filtered.groupby('Amount Cluster')['Amount'].agg(['min', 'max', 'mean']).round(2)
    cluster_labels = {
        0: f"Small (₹{cluster_stats.loc[0, 'min']:,.2f} - ₹{cluster_stats.loc[0, 'max']:,.2f})",
        1: f"Medium (₹{cluster_stats.loc[1, 'min']:,.2f} - ₹{cluster_stats.loc[1, 'max']:,.2f})",
        2: f"Large (₹{cluster_stats.loc[2, 'min']:,.2f} - ₹{cluster_stats.loc[2, 'max']:,.2f})"
    }
    df_filtered['Cluster Label'] = df_filtered['Amount Cluster'].map(cluster_labels)
    
    # Display cluster statistics
    st.write("Cluster Statistics:")
    st.dataframe(cluster_stats.style.format({
        'min': '₹{:.2f}',
        'max': '₹{:.2f}',
        'mean': '₹{:.2f}'
    }))
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Matplotlib Visualization", "Plotly Visualization"])
    
    with tab1:
        # Matplotlib visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        df_filtered.boxplot(column='Amount', by='Cluster Label', ax=ax1)
        ax1.set_title('Transaction Amount Distribution by Cluster')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Amount (₹)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar chart of transaction counts
        cluster_counts = df_filtered['Cluster Label'].value_counts()
        cluster_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Number of Transactions per Cluster')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Transactions')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Plotly visualization
        fig = plot_amount_clusters(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Payees and Recipients Analysis
    st.subheader("Top Payees and Recipients")
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["By Transaction Count", "By Total Amount"])
    
    with tab1:
        # Top payees by transaction count
        top_payees_count = df_filtered[df_filtered['Transaction Type'].isin(['Paid', 'Sent'])].groupby('Payee').agg(
            Transaction_Count=('Amount', 'count'),
            Total_Amount=('Amount', 'sum')
        ).sort_values('Transaction_Count', ascending=False).head(10)
        
        # Top recipients by transaction count
        top_recipients_count = df_filtered[df_filtered['Transaction Type'] == 'Received'].groupby('Payee').agg(
            Transaction_Count=('Amount', 'count'),
            Total_Amount=('Amount', 'sum')
        ).sort_values('Transaction_Count', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Payees by Transaction Count")
            st.dataframe(top_payees_count.style.format({
                'Total_Amount': '₹{:.2f}'
            }))
        with col2:
            st.write("Top Recipients by Transaction Count")
            st.dataframe(top_recipients_count.style.format({
                'Total_Amount': '₹{:.2f}'
            }))
    
    with tab2:
        # Top payees by total amount
        top_payees_amount = df_filtered[df_filtered['Transaction Type'].isin(['Paid', 'Sent'])].groupby('Payee').agg(
            Transaction_Count=('Amount', 'count'),
            Total_Amount=('Amount', 'sum')
        ).sort_values('Total_Amount', ascending=False).head(10)
        
        # Top recipients by total amount
        top_recipients_amount = df_filtered[df_filtered['Transaction Type'] == 'Received'].groupby('Payee').agg(
            Transaction_Count=('Amount', 'count'),
            Total_Amount=('Amount', 'sum')
        ).sort_values('Total_Amount', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Payees by Total Amount")
            st.dataframe(top_payees_amount.style.format({
                'Total_Amount': '₹{:.2f}'
            }))
        with col2:
            st.write("Top Recipients by Total Amount")
            st.dataframe(top_recipients_amount.style.format({
                'Total_Amount': '₹{:.2f}'
            }))
    
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
    