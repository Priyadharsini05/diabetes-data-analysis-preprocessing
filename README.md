# Diabetes Data Analysis and Preprocessing

A Python-based data analysis project focused on exploring diabetes-related clinical and symptom data, examining statistical distributions, identifying feature relationships, detecting outliers, and preparing structured datasets for downstream machine learning workflows.

## Overview

This project implements an exploratory data analysis (EDA) and preprocessing pipeline on a diabetes dataset containing demographic attributes and diabetes-associated symptoms.

Using **Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn**, the notebook investigates dataset structure, categorical distributions, statistical properties, feature correlations, and outlier patterns. It also applies categorical encoding and feature scaling to generate processed datasets suitable for subsequent analytical and machine learning tasks.

## Objectives

- Inspect dataset structure, data types, dimensions, and missing-value indicators.
- Perform row selection, column filtering, and conditional data analysis using Pandas.
- Examine diabetes class distributions and demographic characteristics.
- Analyze symptom patterns using grouping and descriptive statistics.
- Investigate feature distributions, skewness, and kurtosis.
- Evaluate pairwise correlations and covariance between features.
- Remove duplicate records and encode categorical variables.
- Detect statistical outliers using the Interquartile Range (IQR) method.
- Apply standardization and min-max scaling to numerical features.
- Export processed datasets for further analysis.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation, filtering, aggregation, and preprocessing |
| NumPy | Numerical computation and array operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization and correlation heatmaps |
| Scikit-learn | Missing-value imputation and feature scaling |
| Jupyter Notebook | Interactive analysis and experimentation |

## Dataset

The notebook loads a CSV file named `diabetes.csv`.

The dataset contains demographic information, symptom indicators, and a target column named `class`.

Features explored in the notebook include:

- **Demographic:** Age, Gender
- **Symptoms and clinical indicators:** Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, and Obesity
- **Target:** `class`, containing the labels `Positive` and `Negative`

The notebook examines the dataset's dimensions, column names, data types, missing-value indicators, and class distribution before preprocessing.

> **Dataset requirement:** Place `diabetes.csv` in the working directory before executing the notebook. The dataset is not included automatically with the notebook.

## Methodology

### 1. Data Inspection and Exploration

The initial analysis examines the dataset using Pandas operations to understand its structure and contents.

Key operations include:
- Dataset dimensions and column inspection.
- Data type identification.
- Missing-value checks.
- Record inspection using `head()`, `tail()`, and row slicing.
- Positional and label-based indexing with `iloc` and `loc`.

### 2. Exploratory Data Analysis

The notebook explores demographic and symptom-related attributes to understand the composition of the dataset.

Analysis includes:
- Frequency distribution of diabetes outcome classes.
- Percentage distribution of target classes.
- Age-based filtering and sorting.
- Mean age grouped by gender.
- Grouped analysis of symptom combinations.
- Statistical summaries of numerical and encoded features.

### 3. Data Cleaning and Categorical Encoding

Duplicate records are removed before subsequent analysis.

Categorical variables are transformed into numerical representations using explicit binary mappings.

| Category | Encoding |
|---|---|
| Gender | Male = 1, Female = 0 |
| Binary symptom indicators | Yes = 1, No = 0 |
| Target class | Positive = 1, Negative = 0 |

The encoded dataset is exported as `diabetesnew.csv`.

These transformations make the categorical variables compatible with numerical statistical operations and common machine learning preprocessing workflows.

### 4. Statistical Distribution Analysis

Descriptive statistics are calculated to examine the characteristics of the processed data.

The notebook evaluates:
- Mean, standard deviation, and percentile statistics.
- Skewness to characterize distribution asymmetry.
- Kurtosis to examine distribution shape.
- Density plots for selected encoded features.

Additional visualizations help investigate feature distributions and relationships.

### 5. Correlation and Covariance Analysis

Pairwise correlation coefficients and covariance values are calculated to examine relationships between features.

Visualizations include:
- Correlation heatmap.
- Covariance heatmap.
- Pair plots.
- Scatter plots comparing selected variables.

The correlation matrix is exported as `out.csv`.

These analyses help identify potentially associated variables and provide an overview of the relationships within the dataset. Correlation alone does not establish causation.

### 6. Outlier Detection Using IQR

The notebook applies the Interquartile Range method to identify observations outside the conventional statistical bounds.

The quartiles and IQR are calculated as:

\[
IQR = Q_3 - Q_1
\]

The lower and upper bounds are:

\[
L = Q_1 - 1.5 \times IQR
\]

\[
U = Q_3 + 1.5 \times IQR
\]

Observations falling outside these bounds in any selected numerical feature are excluded from the resulting dataset.

The filtered dataset is exported as `diabetesafterpreprocessing1.csv`.

**Important:** This implementation removes entire rows containing an outlier in any selected feature. It does not merely cap or replace individual outlier values.

### 7. Missing-Value Imputation and Feature Scaling

The notebook demonstrates preprocessing operations using Scikit-learn.

**Mean imputation**
- Initializes a `SimpleImputer` with the mean strategy.
- Applies the imputer to the outlier-filtered dataset.

**Standardization**
- Uses `StandardScaler` to transform features toward zero mean and unit variance.
- Generates a standardized numerical array.

**Min-max normalization**
- Uses `MinMaxScaler` to rescale features to the default range of 0 to 1.
- Exports the normalized dataset as `diabetesafterpre.csv`.

The notebook therefore explores multiple feature transformation techniques commonly used when preparing data for machine learning.

> **Implementation note:** The notebook calculates the imputed array but subsequently scales `outlier_out`, rather than the imputed array. Consequently, the demonstrated scaling steps do not incorporate the imputation result. The final min-max scaling is also performed on `outlier_out` directly.

## Generated Artifacts

The notebook writes the following CSV files during execution:

| File | Description |
|---|---|
| `diabetesnew.csv` | Dataset after duplicate removal and categorical encoding |
| `out.csv` | Pairwise correlation matrix |
| `diabetesafterpreprocessing1.csv` | Dataset after IQR-based outlier filtering |
| `diabetesafterpre.csv` | Dataset after min-max feature scaling |

These files are generated by running the corresponding notebook cells.

## Project Structure

```text
diabetes-data-analysis/
├── Diabetes_Analysis.ipynb
├── diabetes.csv
├── diabetesnew.csv                    # Generated
├── out.csv                            # Generated
├── diabetesafterpreprocessing1.csv    # Generated
├── diabetesafterpre.csv               # Generated
└── README.md
```

The generated CSV files are outputs of the notebook and do not need to be committed to the repository unless they are required for reproducibility or demonstration.

## Installation and Execution

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- The source dataset, `diabetes.csv`

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd diabetes-data-analysis
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Add the dataset

Place `diabetes.csv` in the same directory as the notebook.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Diabetes_Analysis.ipynb` and execute the cells sequentially.

## Technical Scope

This project demonstrates practical experience with:

- Exploratory data analysis and statistical profiling.
- Tabular data manipulation using Pandas and NumPy.
- Categorical feature encoding.
- Distribution analysis using skewness and kurtosis.
- Correlation and covariance analysis.
- IQR-based statistical outlier detection.
- Mean-based missing-value imputation.
- Standardization and min-max normalization.
- Data export and preprocessing workflow design.

## Limitations and Future Improvements

Potential extensions to this project include:

- Implement a reproducible preprocessing pipeline using Scikit-learn's `Pipeline` and `ColumnTransformer`.
- Verify missing values and apply imputation before calculating scaling parameters.
- Separate training and test datasets before fitting preprocessing transformations in a predictive workflow.
- Compare alternative outlier-handling strategies instead of automatically removing every affected row.
- Add automated validation for categorical values and unexpected data types.
- Build a diabetes classification pipeline using suitable baseline models such as Logistic Regression, Decision Trees, or Random Forest.
- Evaluate any future predictive model using appropriate metrics, including precision, recall, F1-score, and ROC-AUC.

These extensions would build on the existing analysis without implying that predictive modeling has already been implemented.

## Disclaimer

This repository is intended for educational and analytical purposes. The analysis does not constitute a clinical diagnostic system, and the notebook does not establish that its findings can be used to diagnose diabetes in individuals.
