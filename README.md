# Google Pay Transaction Analyzer

A comprehensive Streamlit-based web application for analyzing Google Pay transaction data from HTML activity files. The app provides detailed insights into transaction patterns, amount distributions, spending habits, and behavioral clustering with intelligent fuzzy normalization.

## ✨ Features

### 🔍 **Core Transaction Analysis**
- **Data Overview**: Total transactions, amounts, date ranges, and key metrics
- **Smart Filtering**: Amount range selection with outlier detection
- **Payee Selection**: Focus on specific payees or view all transactions
- **Advanced Search**: Search across payee, amount, date, and transaction type
- **Transaction Details**: Comprehensive transaction information display

### 📊 **Data Visualization & Clustering**
- **Amount-Based Clustering**: Automatic detection of transaction groups using K-means
- **Interactive Charts**: Violin plots, scatter plots, and time series analysis
- **Time-Based Analysis**: Daily, weekly, and monthly transaction patterns
- **Transaction Trends**: Visual representation of spending patterns over time

### 🎯 **Top Transactions Analysis**
- **Most Paid To**: Top payees by total amount and frequency
- **Most Received From**: Top recipients analysis
- **Highest Transactions**: Largest individual transactions
- **Most Frequent**: Transaction patterns by frequency

### 🧠 **Intelligent Data Processing**
- **Fuzzy Normalization**: Smart grouping of similar payee names and entities
- **Similarity Threshold Control**: Adjustable similarity settings (70-95%)
- **Automatic Text Processing**: Handles variations in names and formatting
- **Memory Management**: Efficient processing of large HTML files

### 🔬 **Behavioral Analysis**
- **Pattern Recognition**: Identify spending behavior patterns
- **Feature Engineering**: Temporal, frequency, and categorical features
- **Machine Learning**: Advanced clustering algorithms for behavior analysis
- **Interactive Results**: Visualize behavioral clusters and patterns

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/google-pay-analyzer.git
cd google-pay-analyzer
```

2. **Create and activate virtual environment:**
```bash
python -m venv gpay
gpay\Scripts\activate  # Windows
# source gpay/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📱 Usage

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Upload your Google Pay activity HTML file**
   - Export from Google Takeout
   - Supports large files with memory management
   - Automatic HTML parsing and data extraction

3. **Explore your data:**
   - **Adjust fuzzy similarity threshold** for name grouping
   - **Filter by amount ranges** to focus on specific transactions
   - **Select payees** for targeted analysis
   - **Search transactions** using multiple criteria
   - **View visualizations** and clustering results
   - **Run behavioral analysis** to discover patterns

## 🛠️ Technical Features

### **Data Processing**
- HTML parsing with BeautifulSoup
- Memory-efficient large file handling
- Automatic data type detection and conversion
- Robust error handling and validation

### **Machine Learning**
- K-means clustering with optimal k detection
- Silhouette score optimization
- Feature scaling and normalization
- Dimensionality reduction with PCA

### **Fuzzy Matching**
- Multiple similarity algorithms (ratio, partial, token-based)
- Unicode normalization and text cleaning
- Configurable similarity thresholds
- Intelligent grouping of similar entities

## 📁 Project Structure

```
google-pay-analyzer/
├── app.py                 # Main Streamlit application with all features
├── try.py                 # Behavioral experiments and testing
├── requirements.txt       # Python dependencies
├── data/                  # Sample data and HTML files
├── scripts/               # Utility scripts and notebooks
├── gpay/                  # Virtual environment
└── README.md             # Project documentation
```

## 📋 Requirements

- **Python 3.8+**
- **Core Libraries**: Streamlit, Pandas, NumPy, Matplotlib
- **ML Libraries**: scikit-learn, Plotly
- **Text Processing**: BeautifulSoup4, fuzzywuzzy, python-Levenshtein
- **Data Handling**: Arrow compatibility for large datasets

## 🔧 Configuration

### **Fuzzy Normalization Settings**
- **Similarity Threshold**: 70-95% (default: 85%)
- **Text Processing**: Automatic unicode normalization
- **Grouping Strategy**: Multi-algorithm similarity matching

### **Clustering Parameters**
- **Amount Clustering**: Automatic optimal k detection
- **Behavioral Clustering**: Advanced feature engineering
- **Visualization**: Interactive plots and charts

## 🎨 User Interface

- **Responsive Design**: Wide layout with organized sections
- **Interactive Controls**: Sliders, dropdowns, and search boxes
- **Real-time Updates**: Dynamic filtering and visualization
- **Tabbed Interface**: Organized analysis sections
- **Progress Indicators**: File processing feedback

## 🚨 Error Handling

- **File Size Validation**: Automatic large file detection
- **Memory Management**: Efficient processing of large datasets
- **Data Validation**: Robust error checking and user feedback
- **Graceful Degradation**: Fallback options for edge cases

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports and feature requests
- Code improvements and optimizations
- Documentation updates
- New analysis features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the web application framework
- **scikit-learn** for machine learning capabilities
- **BeautifulSoup** for HTML parsing
- **fuzzywuzzy** for intelligent text matching
- **Plotly** for interactive visualizations

## 📞 Support

For issues, questions, or contributions:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Include sample data if possible (anonymized)
4. Provide error messages and system information

---

**Built with ❤️ for financial data analysis and pattern recognition** 