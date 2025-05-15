import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set_style("whitegrid")

def create_sample_csv(path='data.csv'):
    """
    Generate a small sample CSV for testing/illustration and exit.
    """
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'age':  [30,      25,    22],
        'city': ['New York', 'San Francisco', 'Chicago']
    })
    df.to_csv(path, index=False)
    print(f"Sample CSV written to '{path}'")
    sys.exit(0)

def load_dataset(csv_path=None):
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV dataset from '{csv_path}'")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at '{csv_path}'", file=sys.stderr)
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: No data in file '{csv_path}'", file=sys.stderr)
            sys.exit(1)
        except pd.errors.ParserError as e:
            print(f"Error parsing '{csv_path}': {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error reading '{csv_path}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        print("Loaded built‑in Iris dataset")
        return df

def explore_and_clean(df):
    print("\n--- Data Exploration ---")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("DataFrame info:")
    print(df.info(), "\n")
    print("Missing values before cleaning:")
    print(df.isna().sum(), "\n")

    # Fill missing values in numeric columns with column means
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print("Missing values after cleaning:")
    print(df.isna().sum(), "\n")
    return df

def basic_analysis(df, category_col=None, numeric_col=None):
    """
    Perform basic descriptive statistics and group-by analysis.
    """
    print("\n--- Basic Data Analysis ---")
    print("Descriptive statistics for numerical columns:")
    print(df.describe(), "\n")

    if category_col is None:
        non_numeric = df.select_dtypes(exclude='number').columns
        category_col = non_numeric[0] if len(non_numeric) > 0 else None

    if numeric_col is None:
        numeric_col = df.select_dtypes(include='number').columns[0]

    if category_col:
        grouped = df.groupby(category_col)[numeric_col].agg(['mean', 'median', 'std'])
        print(f"Grouped statistics of '{numeric_col}' by '{category_col}':")
        print(grouped, "\n")
    else:
        print("No categorical column found for grouping.\n")

    print("Observations:")
    print(f"- DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    if category_col:
        counts = df[category_col].value_counts()
        print(f"- Top categories in '{category_col}':")
        print(counts.head(), "\n")
    print(f"- Overall mean of '{numeric_col}': {df[numeric_col].mean():.2f}\n")

    return category_col, numeric_col

def create_visualizations(df, category_col, numeric_cols):
    """
    Create and display:
      - Line chart (if a date column exists)
      - Bar chart of category counts
      - Histogram of a numeric column
      - Scatter plot of two numeric columns
    """
    # 1. Line chart (if a datetime column exists)
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
    if date_cols.any():
        date_col = date_cols[0]
        df_sorted = df.sort_values(date_col).set_index(date_col)
        plt.figure(figsize=(8, 4))
        df_sorted[numeric_cols[0]].plot()
        plt.title(f"Time Series of {numeric_cols[0]}")
        plt.xlabel(date_col)
        plt.ylabel(numeric_cols[0])
        plt.legend([numeric_cols[0]])
        plt.tight_layout()
        plt.show()

    # 2. Bar chart of category counts
    if category_col:
        plt.figure(figsize=(6, 4))
        df[category_col].value_counts().plot(kind='bar')
        plt.title(f"Count per {category_col}")
        plt.xlabel(category_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # 3. Histogram of the first numeric column
    plt.figure(figsize=(6, 4))
    df[numeric_cols[0]].hist(bins=20)
    plt.title(f"Distribution of {numeric_cols[0]}")
    plt.xlabel(numeric_cols[0])
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 4. Scatter plot between the first two numeric columns (if available)
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7)
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.tight_layout()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Data Analysis Script")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Path to CSV file. If omitted, the Iris dataset is used.",
        default=None
    )
    parser.add_argument(
        "--create-sample",
        dest="create_sample",
        action="store_true",
        help="Generate a sample_data.csv file and exit."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.create_sample:
        create_sample_csv(path='data.csv')

    df = load_dataset(csv_path=args.csv_path)
    df = explore_and_clean(df)
    category_col, numeric_col = basic_analysis(df)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    create_visualizations(df, category_col, numeric_cols)

if __name__ == "__main__":
    main()
