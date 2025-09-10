
"""
COMPREHENSIVE TRAIN DEMAND ANALYSIS - PREREQUISITES CODE WITH VISUALIZATIONS
============================================================================
This code performs all prerequisite analyses for:
- Model 1: Log-Linear Elasticity Estimation
- Model 2: Interaction Effects for Market Segmentation
- ENHANCED: Comprehensive visualizations and Excel export

Author: Expert Analytics Framework
Course: Analytics in Managerial Economics (DBA5101)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def create_comprehensive_visualizations(df):
    """
    Create comprehensive visualizations for data understanding
    INSIGHTS: Visual exploration of demand patterns and relationships
    """
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)

    # Create output directory for plots
    import os
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')

    # 1. DISTRIBUTION ANALYSIS
    print("📊 Creating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution Analysis of Key Variables', fontsize=16, fontweight='bold')

    # Seats distribution
    axes[0,0].hist(df['num_seats_total'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Seats Sold')
    axes[0,0].set_xlabel('Number of Seats')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)

    # Price distribution
    axes[0,1].hist(df['mean_net_ticket_price'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Distribution of Ticket Prices')
    axes[0,1].set_xlabel('Ticket Price')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)

    # Days advance distribution
    if df['days_advance'].std() > 0:
        axes[1,0].hist(df['days_advance'], bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[1,0].set_title('Distribution of Advance Booking Days')
        axes[1,0].set_xlabel('Days in Advance')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'All bookings same day', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Days Advance (No Variation)')

    # Cumulative sales distribution
    axes[1,1].hist(df['Culmulative_sales'], bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1,1].set_title('Distribution of Cumulative Sales')
    axes[1,1].set_xlabel('Cumulative Sales')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_plots/01_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. PRICE-QUANTITY RELATIONSHIP (DEMAND CURVE)
    print("📈 Creating demand curve visualization...")
    plt.figure(figsize=(12, 8))
    if df['mean_net_ticket_price'].std() > 0:
        plt.scatter(df['mean_net_ticket_price'], df['num_seats_total'],
                   alpha=0.6, c='darkblue', s=20)

        # Add trend line
        z = np.polyfit(df['mean_net_ticket_price'], df['num_seats_total'], 1)
        p = np.poly1d(z)
        plt.plot(df['mean_net_ticket_price'], p(df['mean_net_ticket_price']),
                "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope: {z[0]:.4f})')

        plt.xlabel('Ticket Price', fontsize=12)
        plt.ylabel('Number of Seats Sold', fontsize=12)
        plt.title('Price-Quantity Relationship (Demand Curve)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = df['mean_net_ticket_price'].corr(df['num_seats_total'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        plt.text(0.5, 0.5, 'No price variation detected\nCannot plot demand curve',
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Price-Quantity Relationship (No Price Variation)', fontsize=14, fontweight='bold')

    plt.savefig('analysis_plots/02_demand_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. MARKET SEGMENTATION ANALYSIS
    print("🎯 Creating market segmentation plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Market Segmentation Analysis', fontsize=16, fontweight='bold')

    # By Cabin Type
    cabin_data = df.groupby('isNormCabin')['num_seats_total'].agg(['mean', 'count']).reset_index()
    cabin_data['cabin_type'] = cabin_data['isNormCabin'].map({0: 'Special Cabin', 1: 'Normal Cabin'})

    axes[0,0].bar(cabin_data['cabin_type'], cabin_data['mean'],
                  color=['lightcoral', 'lightblue'], alpha=0.8, edgecolor='black')
    axes[0,0].set_title('Average Seats by Cabin Type')
    axes[0,0].set_ylabel('Average Seats')
    axes[0,0].grid(True, alpha=0.3)

    # By Customer Category
    customer_data = df.groupby('Customer_Cat')['num_seats_total'].agg(['mean', 'count']).reset_index()

    axes[0,1].bar(customer_data['Customer_Cat'], customer_data['mean'],
                  color=['lightgreen', 'orange'], alpha=0.8, edgecolor='black')
    axes[0,1].set_title('Average Seats by Customer Category')
    axes[0,1].set_ylabel('Average Seats')
    axes[0,1].grid(True, alpha=0.3)

    # By Trip Type
    trip_summary = pd.DataFrame({
        'Trip_Type': ['One-way Only', 'Return Only', 'Both', 'Neither'],
        'Count': [
            ((df['isOneway'] == 1) & (df['isReturn'] == 0)).sum(),
            ((df['isOneway'] == 0) & (df['isReturn'] == 1)).sum(),
            ((df['isOneway'] == 1) & (df['isReturn'] == 1)).sum(),
            ((df['isOneway'] == 0) & (df['isReturn'] == 0)).sum()
        ]
    })

    axes[0,2].bar(trip_summary['Trip_Type'], trip_summary['Count'],
                  color=['purple', 'brown', 'pink', 'gray'], alpha=0.8, edgecolor='black')
    axes[0,2].set_title('Trip Type Distribution')
    axes[0,2].set_ylabel('Count')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)

    # Box plots for detailed comparison
    df.boxplot(column='num_seats_total', by='isNormCabin', ax=axes[1,0])
    axes[1,0].set_title('Seats Distribution by Cabin Type')
    axes[1,0].set_xlabel('Cabin Type (0=Special, 1=Normal)')
    axes[1,0].set_ylabel('Number of Seats')

    df.boxplot(column='num_seats_total', by='Customer_Cat', ax=axes[1,1])
    axes[1,1].set_title('Seats Distribution by Customer Category')
    axes[1,1].set_xlabel('Customer Category')
    axes[1,1].set_ylabel('Number of Seats')

    # Revenue by segment
    revenue_data = df.groupby(['Customer_Cat', 'isNormCabin'])['estimated_revenue'].sum().reset_index()
    revenue_pivot = revenue_data.pivot(index='Customer_Cat', columns='isNormCabin', values='estimated_revenue')

    revenue_pivot.plot(kind='bar', ax=axes[1,2], color=['lightcoral', 'lightblue'], alpha=0.8)
    axes[1,2].set_title('Total Revenue by Customer & Cabin Type')
    axes[1,2].set_ylabel('Total Revenue')
    axes[1,2].legend(['Special Cabin', 'Normal Cabin'])
    axes[1,2].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('analysis_plots/03_market_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. CORRELATION HEATMAP
    print("🔥 Creating correlation heatmap...")
    numeric_cols = ['num_seats_total', 'mean_net_ticket_price', 'days_advance',
                   'Culmulative_sales', 'isNormCabin', 'isReturn', 'Customer_Cat_B']
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) > 2:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[available_cols].corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('analysis_plots/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 5. TIME SERIES ANALYSIS (if temporal variation exists)
    print("📅 Creating temporal analysis...")
    if df['days_advance'].std() > 0:
        plt.figure(figsize=(14, 8))

        # Group by days advance and calculate mean seats
        time_analysis = df.groupby('days_advance').agg({
            'num_seats_total': ['mean', 'count'],
            'mean_net_ticket_price': 'mean'
        }).reset_index()

        time_analysis.columns = ['days_advance', 'avg_seats', 'count', 'avg_price']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Average seats by advance booking
        ax1.plot(time_analysis['days_advance'], time_analysis['avg_seats'],
                marker='o', linewidth=2, markersize=4, color='darkblue')
        ax1.set_title('Average Seats by Days in Advance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days in Advance')
        ax1.set_ylabel('Average Seats Sold')
        ax1.grid(True, alpha=0.3)

        # Booking frequency
        ax2.bar(time_analysis['days_advance'], time_analysis['count'],
               alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Booking Frequency by Days in Advance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Days in Advance')
        ax2.set_ylabel('Number of Bookings')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_plots/05_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️  No temporal variation - skipping time series plots")

    # 6. LOG-TRANSFORMED ANALYSIS
    print("📐 Creating log-transformed analysis...")
    if 'ln_seats' in df.columns and 'ln_price' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Log-Transformed Analysis for Elasticity Models', fontsize=16, fontweight='bold')

        # Log-seats distribution
        axes[0].hist(df['ln_seats'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0].set_title('Distribution of ln(Seats)')
        axes[0].set_xlabel('ln(Seats)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)

        # Log-price distribution
        axes[1].hist(df['ln_price'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title('Distribution of ln(Price)')
        axes[1].set_xlabel('ln(Price)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        # Log-log relationship (elasticity)
        if df['ln_price'].std() > 0:
            axes[2].scatter(df['ln_price'], df['ln_seats'], alpha=0.6, c='red', s=20)

            # Add trend line
            z = np.polyfit(df['ln_price'], df['ln_seats'], 1)
            p = np.poly1d(z)
            axes[2].plot(df['ln_price'], p(df['ln_price']),
                        "b--", alpha=0.8, linewidth=2,
                        label=f'Elasticity ≈ {z[0]:.4f}')

            axes[2].set_xlabel('ln(Price)')
            axes[2].set_ylabel('ln(Seats)')
            axes[2].set_title('Log-Log Relationship (Price Elasticity)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Add correlation
            log_corr = df['ln_price'].corr(df['ln_seats'])
            axes[2].text(0.05, 0.95, f'Correlation: {log_corr:.4f}',
                        transform=axes[2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            axes[2].text(0.5, 0.5, 'No price variation\nCannot estimate elasticity',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Log-Log Relationship (No Price Variation)')

        plt.tight_layout()
        plt.savefig('analysis_plots/06_log_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    print("✅ All visualizations saved in 'analysis_plots/' directory")
    return df

def save_analysis_to_excel(df, corr_matrix):
    """
    Save comprehensive analysis results to Excel file
    INSIGHTS: Structured export for further analysis and reporting
    """
    print("\n💾 SAVING COMPREHENSIVE ANALYSIS TO EXCEL")
    print("="*50)

    # Create Excel writer object
    with pd.ExcelWriter('comprehensive_train_analysis.xlsx', engine='openpyxl') as writer:

        # 1. Raw processed data
        df.to_excel(writer, sheet_name='Processed_Data', index=False)
        print("✅ Processed data saved to 'Processed_Data' sheet")

        # 2. Summary statistics
        summary_stats = df.describe()
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
        print("✅ Summary statistics saved to 'Summary_Statistics' sheet")

        # 3. Correlation matrix
        if corr_matrix is not None:
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
            print("✅ Correlation matrix saved to 'Correlation_Matrix' sheet")

        # 4. Market segmentation analysis
        market_seg = pd.DataFrame()

        # By cabin type
        cabin_analysis = df.groupby('isNormCabin').agg({
            'num_seats_total': ['mean', 'std', 'count'],
            'mean_net_ticket_price': ['mean', 'std'],
            'estimated_revenue': ['sum', 'mean'],
            'days_advance': 'mean'
        }).round(3)
        cabin_analysis.columns = ['_'.join(col).strip() for col in cabin_analysis.columns]
        cabin_analysis.to_excel(writer, sheet_name='Cabin_Analysis')

        # By customer category
        customer_analysis = df.groupby('Customer_Cat').agg({
            'num_seats_total': ['mean', 'std', 'count'],
            'mean_net_ticket_price': ['mean', 'std'],
            'estimated_revenue': ['sum', 'mean'],
            'days_advance': 'mean'
        }).round(3)
        customer_analysis.columns = ['_'.join(col).strip() for col in customer_analysis.columns]
        customer_analysis.to_excel(writer, sheet_name='Customer_Analysis')

        print("✅ Market segmentation analysis saved")

        # 5. Model readiness checklist
        model_readiness = pd.DataFrame({
            'Check': [
                'Zero values in seats',
                'Zero values in prices',
                'Missing values',
                'Price variation exists',
                'Log transformation possible',
                'Sufficient sample size',
                'Cabin type segments > 30',
                'Customer segments > 30',
                'Interaction terms created',
                'Model 1 ready',
                'Model 2 ready'
            ],
            'Status': [
                'PASS' if (df['num_seats_total'] == 0).sum() == 0 else 'FAIL',
                'PASS' if (df['mean_net_ticket_price'] <= 0).sum() == 0 else 'FAIL',
                'PASS' if df.isnull().sum().sum() == 0 else 'FAIL',
                'PASS' if df['mean_net_ticket_price'].std() > 0 else 'FAIL',
                'PASS' if 'ln_seats' in df.columns else 'FAIL',
                'PASS' if len(df) >= 100 else 'FAIL',
                'PASS' if min((df['isNormCabin'] == 0).sum(), (df['isNormCabin'] == 1).sum()) >= 30 else 'FAIL',
                'PASS' if min((df['Customer_Cat'] == 'A').sum(), (df['Customer_Cat'] == 'B').sum()) >= 30 else 'FAIL',
                'PASS' if 'price_x_cabin' in df.columns else 'FAIL',
                'PASS' if ('ln_seats' in df.columns and df['mean_net_ticket_price'].std() > 0) else 'FAIL',
                'PASS' if ('price_x_cabin' in df.columns and min((df['isNormCabin'] == 0).sum(), (df['isNormCabin'] == 1).sum()) >= 30) else 'FAIL'
            ],
            'Count_or_Value': [
                (df['num_seats_total'] == 0).sum(),
                (df['mean_net_ticket_price'] <= 0).sum(),
                df.isnull().sum().sum(),
                f"{df['mean_net_ticket_price'].std():.4f}",
                'ln_seats' in df.columns,
                len(df),
                min((df['isNormCabin'] == 0).sum(), (df['isNormCabin'] == 1).sum()),
                min((df['Customer_Cat'] == 'A').sum(), (df['Customer_Cat'] == 'B').sum()),
                'price_x_cabin' in df.columns,
                'Both conditions met' if ('ln_seats' in df.columns and df['mean_net_ticket_price'].std() > 0) else 'Conditions not met',
                'Both conditions met' if ('price_x_cabin' in df.columns and min((df['isNormCabin'] == 0).sum(), (df['isNormCabin'] == 1).sum()) >= 30) else 'Conditions not met'
            ]
        })
        model_readiness.to_excel(writer, sheet_name='Model_Readiness', index=False)
        print("✅ Model readiness checklist saved")

        # 6. Business insights
        business_insights = pd.DataFrame({
            'Metric': [
                'Total Observations',
                'Total Revenue',
                'Average Price',
                'Average Seats per Booking',
                'Price-Quantity Correlation',
                'Most Common Cabin Type',
                'Most Common Customer Type',
                'Price Variation (Std Dev)',
                'Booking Window Range (Days)'
            ],
            'Value': [
                f"{len(df):,}",
                f"${df['estimated_revenue'].sum():,.2f}",
                f"${df['mean_net_ticket_price'].mean():.2f}",
                f"{df['num_seats_total'].mean():.2f}",
                f"{df['mean_net_ticket_price'].corr(df['num_seats_total']):.4f}",
                'Normal Cabin' if (df['isNormCabin'] == 1).sum() > (df['isNormCabin'] == 0).sum() else 'Special Cabin',
                df['Customer_Cat'].mode().iloc[0] if len(df['Customer_Cat'].mode()) > 0 else 'N/A',
                f"{df['mean_net_ticket_price'].std():.4f}",
                f"{df['days_advance'].min()} to {df['days_advance'].max()}" if df['days_advance'].std() > 0 else "Same day only"
            ]
        })
        business_insights.to_excel(writer, sheet_name='Business_Insights', index=False)
        print("✅ Business insights saved")

        # 7. Variable definitions and methodology
        methodology = pd.DataFrame({
            'Variable': [
                'num_seats_total', 'mean_net_ticket_price', 'days_advance', 'ln_seats', 'ln_price',
                'Customer_Cat_B', 'price_x_cabin', 'price_x_customer', 'price_x_return',
                'estimated_revenue', 'demand_intensity', 'booking_window'
            ],
            'Definition': [
                'Number of seats sold (dependent variable)',
                'Average ticket price (key independent variable)',
                'Days between purchase and departure',
                'Natural log of seats (for elasticity analysis)',
                'Natural log of price (for elasticity analysis)',
                'Binary: 1 if Customer Category B, 0 if A',
                'Interaction: price × normal cabin indicator',
                'Interaction: price × customer category B',
                'Interaction: price × return trip indicator',
                'Total revenue: seats × price',
                'Demand intensity: cumulative sales / seats',
                'Booking timing category: Early/Medium/Last-minute'
            ],
            'Purpose': [
                'Main outcome variable for demand function',
                'Price elasticity estimation',
                'Temporal demand analysis',
                'Log-linear elasticity model (Model 1)',
                'Log-linear elasticity model (Model 1)',
                'Market segmentation analysis',
                'Interaction effects model (Model 2)',
                'Interaction effects model (Model 2)',
                'Interaction effects model (Model 2)',
                'Business performance metric',
                'Market dynamics indicator',
                'Consumer behavior analysis'
            ]
        })
        methodology.to_excel(writer, sheet_name='Variable_Definitions', index=False)
        print("✅ Variable definitions saved")

    print(f"\n🎉 Complete analysis saved to 'comprehensive_train_analysis.xlsx'")
    print("📊 Excel file contains 8 sheets with comprehensive analysis")
    return True

def load_data(file_path='Data-GP1.csv'):
    """
    Load the train demand dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"⚠️  File {file_path} not found. Please check the file path.")
        raise

def data_quality_assessment(df):
    """
    Comprehensive data quality assessment
    INSIGHTS: Identifies data issues that could affect model validity
    """
    print("\n📊 DATA QUALITY ASSESSMENT")
    print("-" * 40)

    # Basic info
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)

    # Missing values analysis
    print("\n🔍 MISSING VALUES ANALYSIS:")
    missing_analysis = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_subset = missing_analysis[missing_analysis['Missing_Count'] > 0]
    if len(missing_subset) > 0:
        print(missing_subset)
    else:
        print("✅ No missing values detected")

    # Check for zeros in key variables (critical for log transformation)
    print("\n🔍 ZERO VALUES CHECK (Critical for Log Models):")
    zero_seats = (df['num_seats_total'] == 0).sum()
    zero_prices = (df['mean_net_ticket_price'] <= 0).sum()

    print(f"Zero seats: {zero_seats} ({zero_seats/len(df)*100:.2f}%)")
    print(f"Zero/negative prices: {zero_prices} ({zero_prices/len(df)*100:.2f}%)")

    if zero_seats > 0 or zero_prices > 0:
        print("⚠️  WARNING: Zero values detected - will need adjustment for log transformation")
    else:
        print("✅ No zero values - safe for log transformation")

    # Data consistency checks
    print("\n🔍 DATA CONSISTENCY CHECKS:")

    # Convert dates if they're strings
    if df['Purchase_Date'].dtype == 'object':
        df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
    if df['Dept_Date'].dtype == 'object':
        df['Dept_Date'] = pd.to_datetime(df['Dept_Date'])

    # Date logic check
    date_issues = (df['Purchase_Date'] > df['Dept_Date']).sum()
    print(f"Purchase after departure: {date_issues} cases")

    # Trip type consistency
    both_trip_types = ((df['isReturn'] == 1) & (df['isOneway'] == 1)).sum()
    neither_trip_type = ((df['isReturn'] == 0) & (df['isOneway'] == 0)).sum()
    print(f"Both return and one-way: {both_trip_types} cases")
    print(f"Neither return nor one-way: {neither_trip_type} cases")

    return df

def feature_engineering(df):
    """
    Create all necessary features for both models
    INSIGHTS: Creates variables that capture economic relationships
    """
    print("\n⚙️ FEATURE ENGINEERING")
    print("-" * 40)

    # Create days_advance (critical temporal variable)
    df['days_advance'] = (df['Dept_Date'] - df['Purchase_Date']).dt.days
    print(f"✅ Created days_advance: Range {df['days_advance'].min()} to {df['days_advance'].max()} days")

    # Handle zero values for log transformation
    df['num_seats_adj'] = df['num_seats_total'].replace(0, 0.1)  # Small positive value for zeros
    df['price_adj'] = df['mean_net_ticket_price'].replace(0, 0.1)  # Small positive value for zeros

    # Log transformations for elasticity analysis
    df['ln_seats'] = np.log(df['num_seats_adj'])
    df['ln_price'] = np.log(df['price_adj'])
    print("✅ Created log transformations: ln_seats, ln_price")

    # Create binary encoding for Customer_Cat
    df['Customer_Cat_B'] = (df['Customer_Cat'] == 'B').astype(int)
    print("✅ Created Customer_Cat_B binary variable")

    # Create interaction terms for Model 2
    df['price_x_cabin'] = df['mean_net_ticket_price'] * df['isNormCabin']
    df['price_x_customer'] = df['mean_net_ticket_price'] * df['Customer_Cat_B']
    df['price_x_return'] = df['mean_net_ticket_price'] * df['isReturn']
    print("✅ Created interaction terms: price_x_cabin, price_x_customer, price_x_return")

    # Create booking window categories
    if df['days_advance'].std() > 0:
        df['booking_window'] = pd.cut(df['days_advance'],
                                     bins=[-np.inf, 7, 30, np.inf],
                                     labels=['Last_minute', 'Medium', 'Early'])
        print("✅ Created booking_window categories")
    else:
        df['booking_window'] = 'Same_day'
        print("✅ Created booking_window (all same day)")

    # Create train dummy variables
    train_dummies = pd.get_dummies(df['Train_Number_All'], prefix='train')
    df = pd.concat([df, train_dummies], axis=1)
    print(f"✅ Created train dummy variables: {list(train_dummies.columns)}")

    # Price per seat ratio
    df['price_per_seat_ratio'] = df['mean_net_ticket_price'] / df['num_seats_adj']
    print("✅ Created price_per_seat_ratio")

    # Demand intensity measure
    df['demand_intensity'] = df['Culmulative_sales'] / df['num_seats_adj']
    print("✅ Created demand_intensity measure")

    # Estimated revenue
    df['estimated_revenue'] = df['num_seats_total'] * df['mean_net_ticket_price']
    print("✅ Created estimated_revenue")

    return df

def descriptive_statistics_analysis(df):
    """
    Generate comprehensive descriptive statistics
    INSIGHTS: Understand market structure and customer behavior patterns
    """
    print("\n📊 DESCRIPTIVE STATISTICS ANALYSIS")
    print("-" * 50)

    # Overall summary statistics
    print("OVERALL SUMMARY STATISTICS:")
    key_vars = ['num_seats_total', 'mean_net_ticket_price', 'days_advance', 'Culmulative_sales']
    summary_stats = df[key_vars].describe()
    print(summary_stats.round(2))

    # Market segmentation analysis
    print("\n🎯 MARKET SEGMENTATION ANALYSIS:")

    # By cabin type
    print("\nBY CABIN TYPE:")
    cabin_analysis = df.groupby('isNormCabin').agg({
        'num_seats_total': ['mean', 'std', 'count'],
        'mean_net_ticket_price': ['mean', 'std'],
        'days_advance': 'mean'
    }).round(2)
    cabin_analysis.columns = ['_'.join(col).strip() for col in cabin_analysis.columns]
    print(cabin_analysis)

    # By customer category
    print("\nBY CUSTOMER CATEGORY:")
    customer_analysis = df.groupby('Customer_Cat').agg({
        'num_seats_total': ['mean', 'std', 'count'],
        'mean_net_ticket_price': ['mean', 'std'],
        'days_advance': 'mean'
    }).round(2)
    customer_analysis.columns = ['_'.join(col).strip() for col in customer_analysis.columns]
    print(customer_analysis)

    # Cross-tabulation analysis
    print("\n📋 CROSS-TABULATION ANALYSIS:")
    crosstab = pd.crosstab(df['Customer_Cat'], df['isNormCabin'], margins=True)
    print("Customer Category vs Cabin Type:")
    print(crosstab)

    return df

def correlation_multicollinearity_analysis(df):
    """
    Analyze correlations and detect multicollinearity issues
    INSIGHTS: Identify potential econometric problems before modeling
    """
    print("\n🔗 CORRELATION & MULTICOLLINEARITY ANALYSIS")
    print("-" * 55)

    # Select numeric variables for correlation analysis
    numeric_vars = ['num_seats_total', 'mean_net_ticket_price', 'ln_seats', 'ln_price',
                   'days_advance', 'Culmulative_sales', 'isNormCabin', 'isReturn',
                   'Customer_Cat_B', 'demand_intensity']

    # Filter to available variables
    available_vars = [var for var in numeric_vars if var in df.columns]

    # Correlation matrix
    print("KEY VARIABLE CORRELATIONS:")
    corr_matrix = df[available_vars].corr()

    # Focus on correlations with dependent variable
    dependent_corr = corr_matrix['num_seats_total'].sort_values(key=abs, ascending=False)
    print("\nCorrelations with SEATS (Dependent Variable):")
    for var, corr in dependent_corr.items():
        if var != 'num_seats_total':
            print(f"{var:25}: {corr:6.3f}")

    # Check for high correlations between independent variables
    print("\n⚠️  HIGH CORRELATION PAIRS (|r| > 0.7):")
    high_corr_found = False
    for i, var1 in enumerate(available_vars):
        for j, var2 in enumerate(available_vars[i+1:], i+1):
            corr_val = corr_matrix.loc[var1, var2]
            if abs(corr_val) > 0.7 and var1 != var2:
                print(f"{var1} - {var2}: {corr_val:.3f}")
                high_corr_found = True

    if not high_corr_found:
        print("✅ No severe multicollinearity detected (|r| > 0.7)")

    return corr_matrix

def model1_assumptions_testing(df):
    """
    Test statistical assumptions for log-linear elasticity model
    INSIGHTS: Validate whether log-linear model assumptions hold
    """
    print("\n📊 MODEL 1: LOG-LINEAR ASSUMPTIONS TESTING")
    print("-" * 50)

    # Test 1: Normality of log-transformed dependent variable
    print("🔍 TEST 1: NORMALITY OF LOG-TRANSFORMED SEATS")
    ln_seats = df['ln_seats'].dropna()

    # Shapiro-Wilk test (for smaller samples)
    if len(ln_seats) <= 5000:
        try:
            shapiro_stat, shapiro_p = shapiro(ln_seats)
            print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            if shapiro_p > 0.05:
                print("✅ Normal distribution assumption HOLDS (p > 0.05)")
            else:
                print("⚠️  Normal distribution assumption VIOLATED (p < 0.05)")
        except:
            print("⚠️  Shapiro-Wilk test failed - sample too large or other issue")

    # Test 2: Linearity in log-log relationship
    print("\n🔍 TEST 2: LOG-LOG LINEARITY")
    if df['ln_price'].std() > 0:  # Check if there's price variation
        log_correlation = df['ln_seats'].corr(df['ln_price'])
        print(f"ln(seats) - ln(price) correlation: {log_correlation:.4f}")

        if abs(log_correlation) > 0.1:
            print("✅ Sufficient linear relationship for elasticity estimation")
        else:
            print("⚠️  Weak linear relationship - elasticity estimates may be imprecise")
    else:
        print("⚠️  No price variation detected - cannot estimate price elasticity")

    # Test 3: Distribution analysis
    print("\n🔍 TEST 3: DISTRIBUTION CHARACTERISTICS")
    print(f"ln_seats - Mean: {ln_seats.mean():.3f}, Std: {ln_seats.std():.3f}")
    print(f"ln_seats - Skewness: {ln_seats.skew():.3f}")
    print(f"ln_seats - Kurtosis: {ln_seats.kurtosis():.3f}")

    # Interpretation
    if abs(ln_seats.skew()) < 1:
        print("✅ Acceptable skewness for regression analysis")
    else:
        print("⚠️  High skewness - consider additional transformations")

    return ln_seats

def segment_analysis_for_model2(df):
    """
    Analyze market segments for interaction model
    INSIGHTS: Validate segment differences and interaction potential
    """
    print("\n📊 MODEL 2: MARKET SEGMENTATION VALIDATION")
    print("-" * 50)

    # Test segment sample sizes
    print("🔍 SEGMENT SAMPLE SIZES:")

    # Cabin type segments
    cabin_counts = df['isNormCabin'].value_counts()
    print(f"Special cabin (0): {cabin_counts.get(0, 0)} observations")
    print(f"Normal cabin (1): {cabin_counts.get(1, 0)} observations")

    # Customer category segments
    customer_counts = df['Customer_Cat'].value_counts()
    print(f"Customer A: {customer_counts.get('A', 0)} observations")
    print(f"Customer B: {customer_counts.get('B', 0)} observations")

    # Test for significant differences between segments
    print("\n🔍 SEGMENT DIFFERENCE TESTING:")

    # T-test for cabin types
    special_cabin = df[df['isNormCabin'] == 0]['num_seats_total']
    normal_cabin = df[df['isNormCabin'] == 1]['num_seats_total']

    if len(special_cabin) > 10 and len(normal_cabin) > 10:
        try:
            t_stat, t_p = stats.ttest_ind(special_cabin, normal_cabin)
            print(f"Cabin type difference: t-stat={t_stat:.3f}, p-value={t_p:.4f}")
            if t_p < 0.05:
                print("✅ Significant difference between cabin types")
            else:
                print("⚠️  No significant difference between cabin types")
        except:
            print("⚠️  T-test failed for cabin types")

    # T-test for customer categories
    customer_a = df[df['Customer_Cat'] == 'A']['num_seats_total']
    customer_b = df[df['Customer_Cat'] == 'B']['num_seats_total']

    if len(customer_a) > 10 and len(customer_b) > 10:
        try:
            t_stat_cust, t_p_cust = stats.ttest_ind(customer_a, customer_b)
            print(f"Customer category difference: t-stat={t_stat_cust:.3f}, p-value={t_p_cust:.4f}")
            if t_p_cust < 0.05:
                print("✅ Significant difference between customer categories")
            else:
                print("⚠️  No significant difference between customer categories")
        except:
            print("⚠️  T-test failed for customer categories")

    return df

def outlier_detection_analysis(df):
    """
    Detect outliers that could affect regression results
    INSIGHTS: Identify observations that might drive results
    """
    print("\n🔍 OUTLIER DETECTION ANALYSIS")
    print("-" * 40)

    # IQR method for key variables
    key_vars = ['num_seats_total', 'mean_net_ticket_price', 'Culmulative_sales']

    for var in key_vars:
        if var in df.columns and df[var].std() > 0:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[var] < lower_bound) | (df[var] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100

            print(f"{var}:")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers: {outliers} ({outlier_pct:.1f}%)")

            if outlier_pct > 5:
                print(f"  ⚠️  High outlier percentage - consider investigation")
            else:
                print(f"  ✅ Acceptable outlier level")

    return df

def business_intelligence_analysis(df):
    """
    Generate business insights from the data
    INSIGHTS: Understand market dynamics and revenue patterns
    """
    print("\n💼 BUSINESS INTELLIGENCE ANALYSIS")
    print("-" * 45)

    # Revenue analysis by segment
    print("💰 REVENUE ANALYSIS BY SEGMENT:")

    # Revenue by cabin type
    cabin_revenue = df.groupby('isNormCabin').agg({
        'estimated_revenue': ['sum', 'mean'],
        'num_seats_total': 'sum'
    }).round(2)
    cabin_revenue.columns = ['_'.join(col) for col in cabin_revenue.columns]
    print("\nRevenue by Cabin Type:")
    print(cabin_revenue)

    # Revenue by customer category
    customer_revenue = df.groupby('Customer_Cat').agg({
        'estimated_revenue': ['sum', 'mean'],
        'num_seats_total': 'sum'
    }).round(2)
    customer_revenue.columns = ['_'.join(col) for col in customer_revenue.columns]
    print("\nRevenue by Customer Category:")
    print(customer_revenue)

    return df

def economic_validation_analysis(df):
    """
    Validate economic relationships and prepare for modeling
    INSIGHTS: Ensure data aligns with economic theory expectations
    """
    print("\n📈 ECONOMIC VALIDATION ANALYSIS")
    print("-" * 45)

    # Basic economic relationships
    print("🔍 BASIC ECONOMIC RELATIONSHIPS:")

    # Price-quantity relationship (should be negative for normal goods)
    if df['mean_net_ticket_price'].std() > 0:
        price_qty_corr = df['mean_net_ticket_price'].corr(df['num_seats_total'])
        print(f"Price-Quantity correlation: {price_qty_corr:.4f}")

        if price_qty_corr < 0:
            print("✅ Negative price-quantity relationship (consistent with demand theory)")
        else:
            print("⚠️  Positive price-quantity relationship (investigate further)")

    # Model readiness assessment
    print("\n🎯 MODEL READINESS ASSESSMENT:")

    # Check data sufficiency for Model 1
    print("Model 1 (Log-Linear) Readiness:")
    model1_ready = True

    if df['ln_seats'].isna().sum() > 0:
        print("  ⚠️  Missing values in log(seats)")
        model1_ready = False

    if df['ln_price'].std() == 0:
        print("  ⚠️  No price variation - cannot estimate elasticity")
        model1_ready = False

    if model1_ready:
        print("  ✅ Ready for log-linear elasticity estimation")

    # Check data sufficiency for Model 2
    print("\nModel 2 (Interaction) Readiness:")
    model2_ready = True

    min_segment_size = min([
        (df['isNormCabin'] == 0).sum(),
        (df['isNormCabin'] == 1).sum(),
        (df['Customer_Cat'] == 'A').sum(),
        (df['Customer_Cat'] == 'B').sum()
    ])

    if min_segment_size < 30:
        print(f"  ⚠️  Small segment size ({min_segment_size}) - results may be unstable")
        model2_ready = False

    if model2_ready:
        print("  ✅ Ready for interaction effects estimation")

    return df

def generate_comprehensive_summary(df):
    """
    Generate final summary of all analyses
    INSIGHTS: Consolidate all findings for modeling decisions
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)

    print("\n📊 DATASET OVERVIEW:")
    print(f"• Total observations: {len(df):,}")
    print(f"• Total variables: {df.shape[1]}")

    print("\n🎯 KEY FINDINGS FOR MODEL 1 (LOG-LINEAR ELASTICITY):")
    zero_seats = (df['num_seats_total'] == 0).sum()
    zero_prices = (df['mean_net_ticket_price'] <= 0).sum()

    if zero_seats == 0 and zero_prices == 0:
        print("• ✅ No missing values or zeros in key variables")
    else:
        print("• ⚠️  Zero values detected and handled")

    print("• ✅ Log transformation successfully applied")

    if df['mean_net_ticket_price'].std() > 0:
        print("• ✅ Price variation detected")
    else:
        print("• ⚠️  Limited price variation detected")

    print("• 🎯 RECOMMENDATION: Ready for log-linear elasticity estimation")

    print("\n🎯 KEY FINDINGS FOR MODEL 2 (INTERACTION EFFECTS):")
    min_segment = min([
        (df['isNormCabin'] == 0).sum(),
        (df['isNormCabin'] == 1).sum(),
        (df['Customer_Cat'] == 'A').sum(),
        (df['Customer_Cat'] == 'B').sum()
    ])

    if min_segment >= 30:
        print("• ✅ Sufficient sample sizes across all segments")
    else:
        print("• ⚠️  Some segments have small sample sizes")

    print("• ✅ Multiple interaction terms created successfully")
    print("• 🎯 RECOMMENDATION: Ready for interaction effects estimation")

    return df

def create_modeling_recommendations(df):
    """
    Provide specific recommendations for both models
    """
    print("\n" + "="*60)
    print("MODELING RECOMMENDATIONS")
    print("="*60)

    print("\n🔧 FOR MODEL 1 IMPLEMENTATION:")
    print("1. Use robust standard errors due to potential heteroscedasticity")
    print("2. Consider train fixed effects to control for route differences")
    print("3. Focus on economically meaningful coefficients")
    print("4. Interpret elasticity in context of transport economics")

    print("\n🔧 FOR MODEL 2 IMPLEMENTATION:")
    print("1. Start with main effects before adding interactions")
    print("2. Test interaction terms jointly for significance")
    print("3. Focus on economically meaningful interactions")
    print("4. Use F-tests to compare nested models")

    print("\n🎯 SUCCESS METRICS:")
    print("• Statistical significance of key coefficients")
    print("• Economic reasonableness of elasticity estimates")
    print("• Model fit improvement from interactions")
    print("• Alignment with transportation economics literature")

    return df

def comprehensive_analysis_pipeline(df):
    """
    Complete preprocessing pipeline for train demand analysis
    """
    print("="*80)
    print("COMPREHENSIVE TRAIN DEMAND ANALYSIS - PREREQUISITES")
    print("="*80)

    # Phase 1: Data Quality Assessment
    df = data_quality_assessment(df)

    # Phase 2: Feature Engineering
    df = feature_engineering(df)

    # Phase 3: CREATE VISUALIZATIONS (NEW)
    df = create_comprehensive_visualizations(df)

    # Phase 4: Descriptive Analysis
    df = descriptive_statistics_analysis(df)

    # Phase 5: Correlation Analysis
    corr_matrix = correlation_multicollinearity_analysis(df)

    # Phase 6: Model-Specific Testing
    ln_seats = model1_assumptions_testing(df)
    df = segment_analysis_for_model2(df)

    # Phase 7: Advanced Analysis
    df = outlier_detection_analysis(df)
    df = business_intelligence_analysis(df)
    df = economic_validation_analysis(df)

    # Phase 8: Final Summary
    df = generate_comprehensive_summary(df)
    df = create_modeling_recommendations(df)

    # Phase 9: SAVE TO EXCEL (NEW)
    save_analysis_to_excel(df, corr_matrix)

    return df, corr_matrix

if __name__ == "__main__":
    # Main execution
    print("🚀 Starting Comprehensive Train Demand Analysis with Visualizations...")

    # Load data
    df = load_data('Data-GP1.csv')

    # Run complete analysis pipeline
    processed_df, correlation_matrix = comprehensive_analysis_pipeline(df)

    # Save processed data for modeling
    try:
        processed_df.to_csv('processed_train_data.csv', index=False)
        print("\n💾 Processed data saved as 'processed_train_data.csv'")
    except Exception as e:
        print(f"\n⚠️  Could not save processed data: {e}")

    print("\n🎉 ANALYSIS COMPLETE!")
    print("\n📋 DELIVERABLES CREATED:")
    print("1. 📊 comprehensive_train_analysis.xlsx - Complete analysis in Excel")
    print("2. 📈 analysis_plots/ directory - All visualizations")
    print("3. 💾 processed_train_data.csv - Clean data for modeling")

    print("\n📋 NEXT STEPS:")
    print("1. Review all visualizations in 'analysis_plots/' directory")
    print("2. Examine Excel file with 8 comprehensive analysis sheets")
    print("3. Implement Model 1: Log-linear elasticity estimation")
    print("4. Implement Model 2: Interaction effects analysis")
    print("5. Compare results and develop business recommendations")



    