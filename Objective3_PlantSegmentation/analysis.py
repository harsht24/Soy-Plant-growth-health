import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
def load_and_combine_data():
    htn_d = pd.read_csv('HTN_D_Dataset.csv')
    htn_ww = pd.read_csv('HTN_WW_Dataset.csv')
    ltn_d = pd.read_csv('LTN_D_Dataset.csv')
    ltn_ww = pd.read_csv('LTN_WW_Dataset.csv')

    # Add condition labels
    htn_d['Condition'] = 'HTN-D'
    htn_ww['Condition'] = 'HTN-WW'
    ltn_d['Condition'] = 'LTN-D'
    ltn_ww['Condition'] = 'LTN-WW'

    # Combine all data
    combined_data = pd.concat([htn_d, htn_ww, ltn_d, ltn_ww], ignore_index=True)

    # Clean data
    combined_data = combined_data[(combined_data['Weight After [g]'] > 0) & (combined_data['Total Intensity'] > 0)]
    combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])

    return combined_data

def descriptive_stats(data):
    return data.groupby('Condition')[['Normalized Intensity', 'Total Intensity', 'Total Pixels']].describe()

def plot_boxplots(data):
    metrics = ['Normalized Intensity', 'Total Intensity', 'Total Pixels']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Condition', y=metric, data=data)
        plt.title(f'{metric} by Growth Condition')
        plt.ylabel(metric)
        plt.xlabel('Growth Condition')
        plt.savefig(f'{metric}_by_Growth_Condition.png')
        plt.show()

def plot_correlation(data):
    correlation_matrix = data[['Normalized Intensity', 'Total Intensity', 'Total Pixels', 'Weight Before [g]']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('Correlation_Matrix.png')
    plt.show()

def plant_health_variations(data):
    plant_stats = data.groupby('IdTag')['Normalized Intensity'].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(14, 8))
    sns.barplot(data=plant_stats, x='IdTag', y='mean', palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Mean Normalized Intensity for Each Plant')
    plt.ylabel('Mean Normalized Intensity')
    plt.xlabel('Plant ID')
    plt.tight_layout()
    plt.savefig('Mean_Normalized_Intensity_for_Each_Plant.png')
    plt.show()

def temporal_analysis(data, condition_name):
    condition_data = data[data['Condition'] == condition_name]
    temporal_trends = condition_data.groupby([condition_data['Timestamp'].dt.date, 'IdTag'])['Normalized Intensity'].mean().reset_index()
    temporal_trends.columns = ['Date', 'Plant ID', 'Mean Normalized Intensity']

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=temporal_trends, x='Date', y='Mean Normalized Intensity', hue='Plant ID', marker="o")
    plt.title(f'Temporal Analysis of Normalized Intensity for {condition_name} Plants')
    plt.xlabel('Date')
    plt.ylabel('Mean Normalized Intensity')
    plt.legend(title='Plant ID', loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'Temporal_Analysis_{condition_name}.png')
    plt.show()

def plot_combined_trend(data):
    combined_trends = data.groupby([data['Timestamp'].dt.date, 'Condition'])['Normalized Intensity'].mean().reset_index()
    combined_trends.columns = ['Date', 'Condition', 'Mean Normalized Intensity']

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=combined_trends, x='Date', y='Mean Normalized Intensity', hue='Condition', marker="o")
    plt.title('Combined Temporal Trend Across Conditions')
    plt.xlabel('Date')
    plt.ylabel('Mean Normalized Intensity')
    plt.legend(title='Condition', loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Combined_Temporal_Trend_Across_Conditions.png')
    plt.show()

# Execute the comprehensive analysis
all_data = load_and_combine_data()

# Descriptive Statistics
descriptive_statistics = descriptive_stats(all_data)
print(descriptive_statistics)

# Boxplots
plot_boxplots(all_data)

# Correlation Analysis
plot_correlation(all_data)

# Plant-Specific Variations
plant_health_variations(all_data)

# Temporal Analysis for Each Condition
temporal_analysis(all_data, 'HTN-WW')
temporal_analysis(all_data, 'HTN-D')
temporal_analysis(all_data, 'LTN-WW')
temporal_analysis(all_data, 'LTN-D')

# Combined Temporal Trend Across Conditions
plot_combined_trend(all_data)
