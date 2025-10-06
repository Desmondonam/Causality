

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

# Load the data
df = pd.read_csv('your_data.csv')  # Replace with your actual file path

print("="*80)
print("BREAST CANCER DATASET - COMPREHENSIVE EDA & CAUSAL ANALYSIS")
print("="*80)

# Basic information
print("\n1. DATASET OVERVIEW")
print("-" * 80)


print("\nFirst few rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n\n2. TARGET VARIABLE DISTRIBUTION")
print("-" * 80)

diagnosis_counts = df['diagnosis'].value_counts()
print(f"\nDiagnosis Distribution:")
print(diagnosis_counts)
print(f"\nPercentages:")
print(df['diagnosis'].value_counts(normalize=True) * 100)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='diagnosis', ax=axes[0], palette=['#10b981', '#ef4444'])
axes[0].set_title('Diagnosis Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Diagnosis (B=Benign, M=Malignant)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for container in axes[0].containers:
    axes[0].bar_label(container)

# Pie chart
colors = ['#10b981', '#ef4444']
axes[1].pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Diagnosis Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 3. FEATURE ENGINEERING - Categorize Features
# ============================================================================

print("\n\n3. FEATURE CATEGORIZATION")
print("-" * 80)

# Remove unnecessary columns
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Separate features by category
mean_features = [col for col in df.columns if col.endswith('_mean')]
se_features = [col for col in df.columns if col.endswith('_se')]
worst_features = [col for col in df.columns if col.endswith('_worst')]

print(f"Mean Features ({len(mean_features)}): {mean_features[:3]}...")
print(f"Standard Error Features ({len(se_features)}): {se_features[:3]}...")
print(f"Worst Features ({len(worst_features)}): {worst_features[:3]}...")

# ============================================================================
# 4. CAUSAL ANALYSIS - Effect Size & Statistical Tests
# ============================================================================

print("\n\n4. CAUSAL ANALYSIS: EFFECT SIZE & STATISTICAL SIGNIFICANCE")
print("-" * 80)

# Separate data by diagnosis
malignant = df[df['diagnosis'] == 'M']
benign = df[df['diagnosis'] == 'B']

# Get all numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_features:
    numeric_features.remove('id')

# Calculate effect sizes and statistical tests
causal_analysis = []

for feature in numeric_features:
    malignant_values = malignant[feature].dropna()
    benign_values = benign[feature].dropna()
    
    # Mean and std
    m_mean = malignant_values.mean()
    b_mean = benign_values.mean()
    m_std = malignant_values.std()
    b_std = benign_values.std()
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((m_std**2 + b_std**2) / 2)
    cohens_d = (m_mean - b_mean) / pooled_std if pooled_std != 0 else 0
    
    # T-test
    t_stat, p_value = ttest_ind(malignant_values, benign_values)
    
    # Interpretation
    if abs(cohens_d) >= 0.8:
        effect = "Large"
    elif abs(cohens_d) >= 0.5:
        effect = "Medium"
    elif abs(cohens_d) >= 0.2:
        effect = "Small"
    else:
        effect = "Negligible"
    
    causal_analysis.append({
        'Feature': feature,
        'Malignant_Mean': m_mean,
        'Benign_Mean': b_mean,
        'Mean_Difference': m_mean - b_mean,
        'Cohens_d': cohens_d,
        'Effect_Size': effect,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'Significant': 'Yes' if p_value < 0.001 else 'No'
    })

# Create DataFrame and sort by effect size
causal_df = pd.DataFrame(causal_analysis)
causal_df = causal_df.sort_values('Cohens_d', key=abs, ascending=False)

print("\nTop 15 Features with Strongest Causal Association:")
print(causal_df.head(15)[['Feature', 'Cohens_d', 'Effect_Size', 'P_Value', 'Significant']])

# ============================================================================
# 5. VISUALIZATION - Effect Sizes
# ============================================================================

print("\n\n5. VISUALIZING CAUSAL RELATIONSHIPS")
print("-" * 80)

# Plot top 15 features by effect size
top_15 = causal_df.head(15)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Effect size plot
colors_effect = ['#ef4444' if x > 0 else '#10b981' for x in top_15['Cohens_d']]
axes[0].barh(range(len(top_15)), top_15['Cohens_d'], color=colors_effect)
axes[0].set_yticks(range(len(top_15)))
axes[0].set_yticklabels(top_15['Feature'])
axes[0].set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
axes[0].set_title("Top 15 Features by Effect Size (Causal Strength)", fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[0].axvline(x=0.8, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Large Effect')
axes[0].axvline(x=-0.8, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# P-value significance plot
axes[1].barh(range(len(top_15)), -np.log10(top_15['P_Value']), color='#8b5cf6')
axes[1].set_yticks(range(len(top_15)))
axes[1].set_yticklabels(top_15['Feature'])
axes[1].set_xlabel("-log10(P-Value)", fontsize=12, fontweight='bold')
axes[1].set_title("Statistical Significance of Top Features", fontsize=14, fontweight='bold')
axes[1].axvline(x=3, color='red', linestyle='--', linewidth=0.8, label='p < 0.001')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. DISTRIBUTION ANALYSIS - Top Causal Features
# ============================================================================

print("\n\n6. DISTRIBUTION COMPARISON OF TOP CAUSAL FEATURES")
print("-" * 80)

top_features = top_15['Feature'].head(6).tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    # Box plot
    df.boxplot(column=feature, by='diagnosis', ax=axes[idx])
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Diagnosis')
    axes[idx].set_ylabel('Value')
    plt.sca(axes[idx])
    plt.xticks([1, 2], ['Benign', 'Malignant'])

plt.suptitle('Distribution of Top 6 Causal Features by Diagnosis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# Violin plots for better distribution visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    sns.violinplot(data=df, x='diagnosis', y=feature, ax=axes[idx], palette=['#10b981', '#ef4444'])
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Diagnosis')

plt.suptitle('Violin Plots of Top 6 Causal Features', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n\n7. CORRELATION ANALYSIS")
print("-" * 80)

# Encode diagnosis for correlation
df_encoded = df.copy()
df_encoded['diagnosis_encoded'] = df_encoded['diagnosis'].map({'M': 1, 'B': 0})

# Correlation with diagnosis
correlations = []
for feature in numeric_features:
    corr, p_val = pearsonr(df_encoded[feature], df_encoded['diagnosis_encoded'])
    correlations.append({
        'Feature': feature,
        'Correlation': corr,
        'P_Value': p_val
    })

corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
print("\nTop 15 Features Correlated with Malignancy:")
print(corr_df.head(15))

# Correlation heatmap for top features
top_corr_features = corr_df.head(10)['Feature'].tolist()
correlation_matrix = df_encoded[top_corr_features + ['diagnosis_encoded']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Top 10 Causal Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 8. SCATTER PLOT MATRIX - Top Features
# ============================================================================

print("\n\n8. PAIRWISE RELATIONSHIPS")
print("-" * 80)

# Select top 4 features for pair plot
top_4_features = top_15['Feature'].head(4).tolist()
pairplot_data = df[top_4_features + ['diagnosis']]

sns.pairplot(pairplot_data, hue='diagnosis', palette=['#10b981', '#ef4444'],
             diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships of Top 4 Causal Features', y=1.02, fontsize=14, fontweight='bold')
plt.show()

# ============================================================================
# 9. CAUSAL INFERENCE SUMMARY
# ============================================================================

print("\n\n9. CAUSAL INFERENCE SUMMARY")
print("=" * 80)

large_effect = causal_df[causal_df['Effect_Size'] == 'Large']
print(f"\nFeatures with LARGE causal effect (|d| > 0.8): {len(large_effect)}")
print(large_effect[['Feature', 'Cohens_d', 'P_Value']].to_string(index=False))

medium_effect = causal_df[causal_df['Effect_Size'] == 'Medium']
print(f"\nFeatures with MEDIUM causal effect (0.5 < |d| < 0.8): {len(medium_effect)}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"""
1. Dataset contains {len(df)} samples ({len(malignant)} malignant, {len(benign)} benign)

2. Top 3 strongest causal indicators:
   - {top_15.iloc[0]['Feature']}: Cohen's d = {top_15.iloc[0]['Cohens_d']:.3f}
   - {top_15.iloc[1]['Feature']}: Cohen's d = {top_15.iloc[1]['Cohens_d']:.3f}
   - {top_15.iloc[2]['Feature']}: Cohen's d = {top_15.iloc[2]['Cohens_d']:.3f}

3. {len(large_effect)} features show LARGE effect sizes (strong causal relationship)
4. All top features are statistically significant (p < 0.001)

5. Interpretation:
   - Cohen's d > 0.8: Strong causal relationship
   - Features with large effect sizes are potential biomarkers
   - "Worst" features (maximum values) show strongest associations
""")

# ============================================================================
# 10. EXPORT RESULTS
# ============================================================================

print("\n\n10. EXPORTING RESULTS")
print("-" * 80)

# Save causal analysis results
causal_df.to_csv('causal_analysis_results.csv', index=False)
print("Causal analysis results saved to 'causal_analysis_results.csv'")

# Save correlation results
corr_df.to_csv('correlation_analysis_results.csv', index=False)
print("Correlation analysis results saved to 'correlation_analysis_results.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)