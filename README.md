# 📊 Econometric Modeling & Demand Analysis

**Data-Driven Demand Function Estimation for Train Travel | Market Segmentation & Dynamic Pricing Insights**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Econometrics](https://img.shields.io/badge/Econometrics-Analysis-orange)
![Status](https://img.shields.io/badge/Status-Research-informational)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Project Overview

A comprehensive econometric analysis project focused on **demand function estimation** for train travel markets. This research combines classical economic theory with modern data science techniques to uncover pricing elasticities, market segmentation patterns, and revenue optimization opportunities in the transportation sector.

**Core Research Question:** How do price changes, demographics, and travel characteristics influence demand for train services across different market segments?

---

## ✨ Key Features

### 📊 **Econometric Modeling**
- **Demand Function Estimation**: OLS, 2SLS, and robust regression techniques
- **Elasticity Analysis**: Price, income, and cross-price elasticity calculations
- **Statistical Validation**: Hypothesis testing, confidence intervals, and model diagnostics

### 🎯 **Market Segmentation**
- **Customer Clustering**: Behavioral segmentation based on travel patterns
- **Demographic Analysis**: Age, income, and occupation-based demand differences
- **Route-specific Insights**: Inter-city vs. regional travel demand patterns

### 💰 **Dynamic Pricing Insights**
- **Revenue Optimization**: Optimal pricing strategies by segment
- **Demand Forecasting**: Predictive models for different scenarios
- **Policy Implications**: Evidence-based recommendations for pricing policies

---

## 🛠️ Methodology

### Econometric Techniques

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **OLS Regression** | Baseline demand estimation | Statsmodels, Scikit-learn |
| **2SLS / IV Methods** | Addressing endogeneity | Linearmodels package |
| **Robust Standard Errors** | Heteroscedasticity correction | HAC estimators |
| **Fixed Effects Models** | Control for unobserved heterogeneity | Panel data analysis |

### Data Analysis Pipeline

1. **Data Collection & Cleaning**
   - Train travel transaction data
   - Demographic and socioeconomic variables
   - Route characteristics and competition data

2. **Exploratory Data Analysis**
   - Summary statistics and distributions
   - Correlation analysis
   - Visualization of demand patterns

3. **Model Specification**
   - Functional form selection (log-log, semi-log, linear)
   - Variable selection and transformation
   - Interaction terms and non-linearities

4. **Estimation & Testing**
   - Parameter estimation
   - Statistical significance testing
   - Model comparison (AIC, BIC, R²)

5. **Interpretation & Policy Recommendations**
   - Elasticity interpretation
   - Segment-specific strategies
   - Revenue impact simulation

---

## 📈 Key Findings (Expected)

✅ **Price Elasticity**: Quantified demand sensitivity to fare changes
✅ **Market Segments**: Identification of 3-5 distinct traveler groups with unique characteristics
✅ **Revenue Optimization**: Data-driven pricing recommendations for different routes
✅ **Policy Insights**: Evidence for subsidy programs and peak/off-peak pricing

---

## 💻 Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Statsmodels**: Econometric modeling and statistical tests
- **Scikit-learn**: Machine learning for segmentation
- **Linearmodels**: Advanced econometric techniques (IV, panel data)

### Visualization & Reporting
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive demand curves and elasticity charts
- **Jupyter Notebooks**: Analysis documentation and reproducibility

---

## 📂 Project Structure (Planned)

```
econometric-modeling-demand-analysis/
│
├── data/
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned and transformed data
│   └── external/                 # Auxiliary data sources
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_demand_estimation.ipynb
│   ├── 04_market_segmentation.ipynb
│   └── 05_pricing_analysis.ipynb
│
├── src/
│   ├── data_processing.py        # Data cleaning utilities
│   ├── econometric_models.py     # Regression functions
│   ├── segmentation.py           # Clustering algorithms
│   └── visualization.py          # Plotting functions
│
├── results/
│   ├── figures/                  # Generated plots
│   ├── tables/                   # Regression output tables
│   └── reports/                  # Summary documents
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started (Planned)

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Basic understanding of econometrics and regression analysis

### Installation

```bash
# Clone the repository
git clone https://github.com/om-gorakhia/econometric-modeling-demand-analysis.git
cd econometric-modeling-demand-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 📚 Research Context

### Economic Theory Foundation
This project builds on established economic principles:
- **Demand Theory**: Downward-sloping demand curves, consumer surplus
- **Price Discrimination**: Segmentation and differential pricing
- **Market Structure**: Competition effects on pricing power

### Applications
- **Transportation Economics**: Train travel demand analysis
- **Revenue Management**: Airline/rail yield optimization techniques
- **Public Policy**: Subsidy design and fare regulation

---

## 🎓 Skills Demonstrated

- **Econometric Analysis**: Regression modeling, causality, and inference
- **Statistical Programming**: Python for data science and economics
- **Market Research**: Segmentation and consumer behavior analysis
- **Business Strategy**: Pricing optimization and revenue management
- **Data Storytelling**: Translating complex analysis into actionable insights

---

## 📄 Academic Applications

This project demonstrates expertise relevant to:
- **Economics Research**: Empirical analysis and hypothesis testing
- **Business Analytics**: Data-driven decision making
- **Consulting**: Market analysis and pricing strategy
- **Policy Analysis**: Evidence-based transportation policy

---

## 🔮 Future Enhancements

- [ ] Panel data analysis with fixed/random effects
- [ ] Machine learning demand prediction models
- [ ] Interactive dashboard for pricing simulations
- [ ] Integration with real-time pricing APIs
- [ ] A/B testing framework for pricing experiments

---

## 📝 License

MIT License - Educational and research purposes.

---

## 👤 Author

**Om Gorakhia**
🎓 NUS MSBA Student | Economics & Analytics Enthusiast | Sustainability Advocate

---

## 🔗 Related Projects

- **Ecommerce Warehouse Optimization**: Supply chain analytics
- **Loan Default Prediction**: Credit risk modeling
- **Tour de France Analytics**: Sports performance analysis

---

## 💬 Contact

For questions about methodology, data sources, or collaboration opportunities:

- **GitHub**: [@om-gorakhia](https://github.com/om-gorakhia)
- **LinkedIn**: [Connect with me](https://www.linkedin.com/in/om-gorakhia)
- **Email**: [e1519898@u.nus.edu](mailto:e1519898@u.nus.edu)

---

**📊 Econometric Modeling & Demand Analysis** | Bridging Economic Theory and Data Science
