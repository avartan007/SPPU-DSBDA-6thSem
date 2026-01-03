# 📊 DSBDA Practicals - Data Science & Big Data Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A comprehensive collection of **Data Science & Big Data Analytics** practicals for 6th Semester Computer Engineering students at **Savitribai Phule Pune University (SPPU)**.

[View Practicals](#practicals) • [Installation](#installation) • [Contributing](#contributing)

</div>

---

## 📚 About DSBDA

**Data Science & Big Data Analytics (DSBDA)** is a core course for 6th-semester Computer Engineering students at SPPU University. This repository contains practical implementations covering essential concepts in:

- 🔄 Data Wrangling & Preprocessing
- 📈 Statistical Analysis
- 🤖 Machine Learning Models
- 🌐 Social Network Analysis
- 📝 Natural Language Processing
- 📊 Data Visualization
- 🧠 Classification Algorithms

---

## 📋 Practicals

### 1. **Data Wrangling I** - Titanic Dataset
Learn data cleaning and preprocessing techniques using the famous Titanic dataset.
- ✅ Data import and exploration
- ✅ Missing value handling
- ✅ Data type conversion
- ✅ Feature encoding & transformation

### 2. **Data Wrangling II** - Student Performance Dataset
Master advanced wrangling techniques including outlier detection and transformation.
- ✅ Outlier detection using IQR method
- ✅ Box plot visualization
- ✅ Log transformation for distribution normalization
- ✅ Data normalization techniques

### 3. **Statistical Analysis** - Iris Dataset
Explore descriptive statistics and grouping operations.
- ✅ Summary statistics & aggregation
- ✅ Group-by operations
- ✅ Descriptive analysis by class
- ✅ Statistical insights

### 4. **Linear Regression** - Boston Housing Dataset
Implement linear regression for house price prediction.
- ✅ Correlation analysis & heatmaps
- ✅ Train-test split & model training
- ✅ Performance metrics (MSE, R² score)
- ✅ Actual vs Predicted visualization

### 5. **Logistic Regression** - Social Network Ads Dataset
Build a classification model for customer purchase prediction.
- ✅ Data preprocessing & feature scaling
- ✅ Logistic regression implementation
- ✅ Confusion matrix & model evaluation
- ✅ Performance metrics (Accuracy, Precision, Recall)

### 6. **Naive Bayes Classification** - Iris Dataset
Implement Gaussian Naive Bayes classifier.
- ✅ Model training & prediction
- ✅ Confusion matrix visualization
- ✅ Accuracy, Precision & Recall calculation
- ✅ Performance analysis

### 7. **Natural Language Processing** - Text Analysis
Explore NLP fundamentals and text processing.
- ✅ Tokenization & POS tagging
- ✅ Stop word removal & stemming
- ✅ Lemmatization techniques
- ✅ TF-IDF vectorization

### 8. **Data Visualization I** - Titanic Dataset
Create informative visualizations for exploratory data analysis.
- ✅ Count plots & distribution analysis
- ✅ Box plots for outlier detection
- ✅ Scatter plots & pair plots
- ✅ Correlation heatmaps

### 9. **Data Visualization II** - Titanic Dataset (Advanced)
Advanced visualization techniques for multivariate analysis.
- ✅ Distribution plots with Hue encoding
- ✅ Categorical vs numerical relationships
- ✅ Multi-dimensional visualization
- ✅ Statistical plot styling

### 10. **Data Visualization III** - Iris Dataset
Comprehensive visualization suite for flower classification.
- ✅ Histogram distributions with KDE
- ✅ Box plots by species
- ✅ Distribution analysis
- ✅ Feature comparison plots

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python & data structures

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ds.git
cd ds
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

---

## 📦 Dependencies

Key libraries used in these practicals:

| Library | Purpose |
|---------|---------|
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | Machine learning algorithms |
| **NLTK** | Natural language processing |

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

---

## 📖 Usage

Each practical is organized as a separate Jupyter notebook. To run a specific practical:

1. Open Jupyter Notebook from the command line
2. Navigate to the desired `.ipynb` file
3. Run cells sequentially (Shift + Enter)
4. Modify code and experiment with parameters

**Example**: To run Data Wrangling I:
```bash
jupyter notebook 1Wrangling.ipynb
```

---

## 📂 Project Structure

```
ds/
├── 1Wrangling.ipynb              # Data Wrangling I
├── 2Wrangling.ipynb              # Data Wrangling II
├── 3Statistics.ipynb             # Statistical Analysis
├── 4LinearRegression.ipynb       # Linear Regression
├── 5SocialNetworks.ipynb         # Logistic Regression
├── 6NaiveBayes.ipynb             # Naive Bayes Classification
├── 7TextAnalysis.ipynb           # NLP Text Analysis
├── 8DataVisualizationI.ipynb     # Data Visualization I
├── 9DataVisualizationII.ipynb    # Data Visualization II
├── 10DataVisualizationIII.ipynb  # Data Visualization III
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🎯 Learning Outcomes

Upon completing these practicals, you will:

✨ Master data wrangling techniques for real-world datasets  
✨ Understand statistical methods for data analysis  
✨ Implement machine learning algorithms from scratch  
✨ Create professional data visualizations  
✨ Process and analyze natural language text  
✨ Build predictive models for classification problems  
✨ Handle missing values, outliers, and data transformations  

---

## 💡 Tips for Learning

- **Start sequentially**: Begin with Data Wrangling to understand preprocessing
- **Experiment**: Modify code and parameters to see how they affect results
- **Visualize**: Always try to visualize your data before and after transformations
- **Document**: Add comments to understand each step
- **Practice**: Try these techniques on your own datasets

---

## 📊 Datasets Used

| Practical | Dataset | Rows | Columns | Source |
|-----------|---------|------|---------|--------|
| Wrangling I, Viz I, II | Titanic | 891 | 12 | Kaggle |
| Wrangling II | Student Performance | 1000 | 8 | Kaggle |
| Statistics, Naive Bayes, Viz III | Iris | 150 | 5 | UCI ML Repository |
| Linear Regression | Boston Housing | 506 | 13 | UCI ML Repository |
| Logistic Regression | Social Network Ads | 400 | 4 | Kaggle |

---

## 🤝 Contributing

Contributions are welcome! If you find any issues or want to enhance these practicals:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 Notes

- All notebooks are designed for **educational purposes**
- Code follows Python best practices and PEP 8 style guide
- Each practical is independent and can be studied in any order (though sequential is recommended)
- External datasets are loaded from URLs where possible; some may require local CSV files

---

## 🎓 University Information

- **University**: Savitribai Phule Pune University (SPPU)
- **Course**: Data Science & Big Data Analytics (DSBDA)
- **Semester**: 6th Semester
- **Branch**: Computer Engineering

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- SPPU University for the course curriculum
- Kaggle and UCI ML Repository for datasets
- The open-source community for amazing libraries

---

## ❓ FAQ

**Q: Can I use these practicals for my own learning?**  
A: Absolutely! This repository is designed for educational purposes.

**Q: Do I need prior ML experience?**  
A: Basic Python knowledge is helpful, but the practicals are beginner-friendly.

**Q: Where can I find the datasets?**  
A: Most datasets are loaded from URLs. Some may need to be downloaded separately.

**Q: Can I contribute improvements?**  
A: Yes! Check the Contributing section above.

---

<div align="center">

⭐ **If you find this helpful, please consider starring the repository!** ⭐

Made with ❤️ for SPPU DSBDA Students

</div>