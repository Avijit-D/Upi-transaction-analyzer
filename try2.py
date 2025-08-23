import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re

def parse_html_to_dataframe(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

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

    df2 = pd.DataFrame(transactions, columns=['Transaction Type', 'Payee', 'Amount', 'Account', 'Date', 'Time'])
    df2['Amount'] = df2['Amount'].str.replace(',', '').astype(float)
    return df2

file_path = r'My Activity.html'
df2 = parse_html_to_dataframe(file_path)
df2.to_csv('extracted_transactions.csv', index=False)

df2['Time'] = df2['Time'].str[:-4]
df2['Time'] = pd.to_datetime(df2['Time'], format='%H:%M:%S')
df2['Date'] = pd.to_datetime(df2['Date'], format='mixed')

print(df2['Time'].dtypes)
print(df2.describe())
print(df2.sample(10))

df = df2.copy()

null_counts = df.isnull().sum()
print(null_counts)

df['Hour'] = df['Time'].dt.hour
hourly_transactions = df.groupby(['Hour', 'Transaction Type']).size().reset_index(name='Count')
pivot_df = hourly_transactions.pivot(index="Hour", columns="Transaction Type", values="Count").fillna(0)

plt.figure(figsize=(12, 6))
plt.bar(pivot_df.index, pivot_df["Paid"], label="Paid", color="blue")
plt.bar(pivot_df.index, pivot_df["Sent"], bottom=pivot_df["Paid"], label="Sent", color="green")
plt.title('Transactions by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.xticks(range(24))
plt.legend(title='Transaction Type')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(pivot_df.index, pivot_df["Received"], label="Received", color="purple")
plt.title('Transactions by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.xticks(range(24))
plt.legend(title='Transaction Type')
plt.show()

df['Day'] = df['Date'].dt.day
day_transactions = df.groupby('Day').agg(
    Count=('Payee', 'size'),
    Amount=('Amount', 'sum')
).reset_index()
print(day_transactions)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(day_transactions['Day'], day_transactions['Count'], color='skyblue')
plt.xlabel('Day')
plt.xticks(range(1, 32))
plt.ylabel('Number of Transactions')
plt.title('Transactions by Day')

ax2 = ax1.twinx()
ax2.plot(day_transactions['Day'], day_transactions['Amount']/day_transactions['Count'])
plt.show()

df['Month'] = df['Date'].dt.month
monthly_transactions = df.groupby('Month').size().reset_index(name='Count')
print(monthly_transactions)

plt.figure(figsize=(12, 6))
plt.bar(monthly_transactions['Month'], monthly_transactions['Count'], color='pink')
plt.xlabel('Month')
tick_positions = range(1, 13)
tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.ylabel('Number of Transactions')
plt.title('Transactions by Month')
plt.show()

amount_ceil = 500
amount_floor = 0
df_limited = df[(df['Amount'] < amount_ceil) & (df['Amount'] > amount_floor)]
plt.figure(figsize=(22, 6))
plt.plot(df_limited['Date'], df_limited['Amount'])
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
transaction_count = [120, 150, 90, 180, 220, 130, 160]
total_amount = [5000, 7000, 4500, 8500, 9000, 6000, 7500]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(days, transaction_count, color='blue', alpha=0.6, label='Transaction Count')
ax1.set_xlabel('Day')
ax1.set_ylabel('Transaction Count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(days, total_amount, color='red', marker='o', linestyle='-', linewidth=2, label='Total Amount')
ax2.set_ylabel('Total Amount', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Transactions per Day with Total Amount Overlay')
plt.show()
