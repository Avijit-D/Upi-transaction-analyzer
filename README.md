# UPI Transaction Analyzer

A Streamlit-based web application for analyzing UPI transaction data from Google Pay activity HTML files. The app provides detailed insights into transaction patterns, amount distributions, and spending habits.

## Features

- **Transaction Analysis**
  - View transaction metrics (total, average, date range)
  - Filter transactions by amount range
  - Search transactions by payee, amount, date, or type
  - Select specific payees for focused analysis

- **Data Visualization**
  - Transaction amount clustering (small, medium, large)
  - Interactive amount range selection
  - Visual representation of transaction patterns
  - Multiple visualization types (Matplotlib and Plotly)

- **Transaction Details**
  - Detailed transaction search functionality
  - Payee-specific analysis
  - Amount-based filtering
  - Date and time-based organization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/upi-transaction-analyzer.git
cd upi-transaction-analyzer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

2. Upload your Google Pay activity HTML file
3. Use the interactive features to analyze your transactions:
   - Adjust the amount range slider
   - Select specific payees
   - Search for specific transactions
   - View various visualizations

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Matplotlib
- NumPy
- BeautifulSoup4
- scikit-learn
- Plotly

## Project Structure

```
upi-transaction-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses scikit-learn for transaction clustering
- BeautifulSoup for HTML parsing 