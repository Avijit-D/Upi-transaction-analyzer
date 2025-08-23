from bs4 import BeautifulSoup
import pandas as pd
import re

def parse_html_to_dataframe(file_path):
    # Load the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    transactions = []
    
    # Find all divs with the relevant class
    for transaction in soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1'):
        # Extract the full text from the div
        transaction_text = transaction.get_text(separator=" ").strip()
        
        # Print the transaction text for debugging
        print("Transaction text:", transaction_text)

        # Regex patterns for different transaction types
        paid_pattern = r'Paid ₹([\d,]+\.\d{2}) to (.+?) using Bank Account (XXXXXX\d{4})'
        sent_pattern = r'Sent ₹([\d,]+\.\d{2}) using Bank Account (XXXXXX\d{4})'
        received_pattern = r'Received ₹([\d,]+\.\d{2})'

        # Attempt to extract the date and time using regex
        date_time_pattern = r'(\d{1,2} \w+ \d{4}, \d{1,2}:\d{2}:\d{2} \w{3})'
        date_time_match = re.search(date_time_pattern, transaction_text)

        if date_time_match:
            date_time = date_time_match.group(1)
            transaction_text = transaction_text[:date_time_match.start()].strip()  # Remove date from the main text for further parsing
            print("Date and time found:", date_time)

            # Split the date_time into separate components
            date_str, time_str = date_time.split(', ')
            time_str = time_str.strip()  # Clean up the time string
        else:
            print("No date found for:", transaction_text)
            continue  # Skip this transaction if no date is found

        # Check for Paid transactions
        match = re.search(paid_pattern, transaction_text)
        if match:
            amount = match.group(1)
            payee = match.group(2)
            account = match.group(3)
            transactions.append(['Paid', payee, amount, account, date_str, time_str])
            continue
        
        # Check for Sent transactions
        match = re.search(sent_pattern, transaction_text)
        if match:
            amount = match.group(1)
            account = match.group(2)
            transactions.append(['Sent', 'N/A', amount, account, date_str, time_str])
            continue
        
        # Check for Received transactions
        match = re.search(received_pattern, transaction_text)
        if match:
            amount = match.group(1)
            transactions.append(['Received', 'N/A', amount, 'N/A', date_str, time_str])
            continue

        print("No match found for:", transaction_text)  # Debug unmatched transaction

    # Create a Pandas DataFrame
    df = pd.DataFrame(transactions, columns=['Transaction Type', 'Payee', 'Amount', 'Account', 'Date', 'Time'])
    
    # Convert amount to a numeric value (removing commas)
    df['Amount'] = df['Amount'].str.replace(',', '').astype(float)
    
    # Check for any NaT in the Date and Time columns
    if df['Date'].isnull().any() or df['Time'].isnull().any():
        print("Warning: Some date or time entries could not be converted:")
        print(df[df['Date'].isnull() | df['Time'].isnull()])

    return df

# Use the function to parse your file
file_path = r'C:\Users\HP\OneDrive\Desktop\Coding\Gpay\data\My Activity.html'

df = parse_html_to_dataframe(file_path)

# Print the DataFrame to check if values are populated
print(df)

# Save the data to a CSV file for further use
df.to_csv('data/extracted_transactions.csv', index=False)
