# Quick Analysis Examples

## Loading Aggregated Data

```python
import pandas as pd
import json

# Load CSV (easiest for pandas)
df = pd.read_csv('results/aggregated_results.csv')

# Or load JSON (preserves nested structures)
with open('results/aggregated_results.json', 'r') as f:
    results = json.load(f)
    df = pd.DataFrame(results)
```

## Scratchpad Analysis

### Who Uses Scratchpad?
```python
# Percentage that used scratchpad
print(f"Scratchpad usage: {df['scratchpad_used'].mean():.1%}")

# Usage by model
print(df.groupby('params.model.name')['scratchpad_used'].mean())
```

### Scratchpad Impact on Crashes
```python
# Compare crash rates
with_scratch = df[df['scratchpad_used'] == True]
without_scratch = df[df['scratchpad_used'] == False]

print(f"Crash rate (with scratchpad): {with_scratch['crashed'].mean():.1%}")
print(f"Crash rate (no scratchpad): {without_scratch['crashed'].mean():.1%}")

# Statistical test
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['scratchpad_used'], df['crashed'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test: p={p_value:.4f}")
```

### Scratchpad Usage Patterns
```python
# Among users, how much do they write?
users = df[df['scratchpad_used'] == True]
print(users[['scratchpad_writes', 'scratchpad_reads', 'scratchpad_deletes']].describe())

# Read/write ratio
users['read_write_ratio'] = users['scratchpad_reads'] / users['scratchpad_writes']
print(f"Average read/write ratio: {users['read_write_ratio'].mean():.2f}")
```

### Examining Scratchpad Contents
```python
# Load a specific run's detailed logs
import json

with open('logs/run_1234567_42/summary.json', 'r') as f:
    summary = json.load(f)
    print("Final scratchpad:")
    for key, value in summary['final_scratchpad'].items():
        print(f"  {key}: {value}")

# Load step-by-step to see evolution
steps = []
with open('logs/run_1234567_42/steps.jsonl', 'r') as f:
    for line in f:
        steps.append(json.loads(line))

# Track scratchpad changes
for step in steps:
    scratch = step['state']['scratchpad']
    if scratch:
        print(f"Step {step['step']}: {scratch}")
```

## Action Pattern Analysis

### Looping Detection
```python
# Runs with high repeat streaks
loopers = df[df['max_repeat_streak'] >= 10]
print(f"Potential loopers: {len(loopers)} ({len(loopers)/len(df):.1%})")

# Most common action among loopers
print(loopers['most_common_action'].value_counts())
```

### Action Diversity
```python
# Average number of unique tools used
print(f"Mean action diversity: {df['action_diversity'].mean():.1f} tools")

# By crash status
print(df.groupby('crashed')['action_diversity'].mean())

# Correlation with survival
import scipy.stats as stats
crashed_only = df[df['crashed'] == True]
r, p = stats.pearsonr(crashed_only['action_diversity'], crashed_only['total_steps'])
print(f"Correlation (diversity vs survival): r={r:.3f}, p={p:.4f}")
```

### Tool Preferences
```python
# Most common actions across all runs
from collections import Counter
all_actions = df['most_common_action'].tolist()
action_counts = Counter(all_actions)
print(action_counts.most_common(10))

# By model
for model in df['params.model.name'].unique():
    model_df = df[df['params.model.name'] == model]
    print(f"\n{model}:")
    print(model_df['most_common_action'].value_counts().head(3))
```

## Budget Analysis

### Bankruptcy Rates
```python
# Overall bankruptcy rate
print(f"Bankruptcy rate: {df['went_bankrupt'].mean():.1%}")

# By p_shock level
print(df.groupby('params.pe_induction.p_shock')['went_bankrupt'].mean())

# Time to bankruptcy
bankrupt = df[df['went_bankrupt'] == True]
print(f"Mean steps to bankruptcy: {bankrupt['total_steps'].mean():.1f}")
```

### Budget Burn Rates
```python
# Distribution of burn rates
print(df['budget_burn_rate'].describe())

# By crash type
print(df.groupby('crash_type')['budget_burn_rate'].mean().sort_values())

# Scatter plot
import matplotlib.pyplot as plt
plt.scatter(df['total_steps'], df['budget_burn_rate'], 
            c=df['crashed'], cmap='RdYlGn_r', alpha=0.6)
plt.xlabel('Total Steps')
plt.ylabel('Budget Burn Rate ($/day)')
plt.title('Budget Burn vs Survival')
plt.colorbar(label='Crashed')
plt.show()
```

### Budget Trajectory
```python
# Minimum budget reached
print(f"Mean minimum budget: ${df['budget_min'].mean():.2f}")

# Who went into debt?
in_debt = df[df['budget_min'] < 0]
print(f"Ran into debt: {len(in_debt)}/{len(df)} ({len(in_debt)/len(df):.1%})")

# Recovery from debt
recovered = in_debt[in_debt['budget_final'] > 0]
print(f"Recovered from debt: {len(recovered)}/{len(in_debt)} ({len(recovered)/len(in_debt):.1%})")
```

## Prediction Error Analysis

### PE Accumulation by Crash Type
```python
# Final PE levels by crash outcome
print(df.groupby('crashed')[['pe_causal_fast_final', 'pe_causal_med_final']].mean())

# By specific crash type
print(df.groupby('crash_type')[['pe_causal_fast_final', 'pe_causal_med_final']].mean())
```

### PE Trajectories
```python
# Load step logs to see PE evolution
steps = []
with open('logs/run_1234567_42/steps.jsonl', 'r') as f:
    for line in f:
        steps.append(json.loads(line))

# Extract PE timeseries
pe_fast = [s['cumulative_pe']['causal']['fast'] for s in steps]
pe_med = [s['cumulative_pe']['causal']['med'] for s in steps]
pe_slow = [s['cumulative_pe']['causal']['slow'] for s in steps]

# Plot
plt.plot(pe_fast, label='Fast EWMA')
plt.plot(pe_med, label='Medium EWMA')
plt.plot(pe_slow, label='Slow EWMA')
plt.xlabel('Step')
plt.ylabel('Cumulative PE')
plt.title('PE Accumulation Over Time')
plt.legend()
plt.show()
```

## Combined Analysis

### Predicting Crashes
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Features
features = [
    'scratchpad_max_size', 'scratchpad_writes', 'scratchpad_reads',
    'max_repeat_streak', 'action_diversity',
    'budget_burn_rate', 'budget_min',
    'pe_causal_fast_final', 'pe_causal_med_final'
]

X = df[features].fillna(0)
y = df['crashed']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

# Accuracy
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Clustering Behaviors
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select behavioral features
behavior_features = [
    'scratchpad_writes', 'max_repeat_streak', 'action_diversity',
    'budget_burn_rate', 'total_steps'
]

X = df[behavior_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Describe clusters
for i in range(4):
    cluster_df = df[df['cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster_df)}):")
    print(cluster_df[behavior_features + ['crashed']].mean())
```

## Visualization Examples

### Scratchpad Heatmap
```python
import seaborn as sns

# Create matrix of scratchpad usage by model and p_shock
pivot = df.pivot_table(
    values='scratchpad_used',
    index='params.model.name',
    columns='params.pe_induction.p_shock',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd')
plt.title('Scratchpad Usage Rate by Model and Shock Level')
plt.show()
```

### Survival Curves
```python
from lifelines import KaplanMeierFitter

# Prepare data
df['event'] = df['crashed'].astype(int)
df['duration'] = df['total_steps']

# Fit KM curves by scratchpad usage
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 6))

for name, group in df.groupby('scratchpad_used'):
    kmf.fit(group['duration'], group['event'], label=f"Scratchpad: {name}")
    kmf.plot_survival_function(ax=ax)

plt.xlabel('Steps')
plt.ylabel('Survival Probability')
plt.title('Survival Analysis: Scratchpad Users vs Non-Users')
plt.show()
```

### Budget Evolution
```python
# Load multiple runs and plot budget trajectories
import matplotlib.pyplot as plt
import glob

fig, ax = plt.subplots(figsize=(12, 6))

for log_file in glob.glob('logs/run_*/steps.jsonl')[:20]:  # First 20 runs
    steps = []
    with open(log_file, 'r') as f:
        for line in f:
            steps.append(json.loads(line))
    
    days = [s['state']['day'] for s in steps]
    budgets = [s['state']['budget'] for s in steps]
    crashed = any(s.get('crashed', False) for s in steps)
    
    ax.plot(days, budgets, alpha=0.3, color='red' if crashed else 'green')

ax.axhline(0, color='black', linestyle='--', label='Bankruptcy Line')
ax.set_xlabel('Day')
ax.set_ylabel('Budget ($)')
ax.set_title('Budget Trajectories (Red=Crashed, Green=Survived)')
ax.legend()
plt.show()
```

## Export for External Tools

### Export to Excel with Multiple Sheets
```python
with pd.ExcelWriter('results/vendomini_analysis.xlsx') as writer:
    # Main results
    df.to_excel(writer, sheet_name='All Results', index=False)
    
    # Scratchpad users only
    df[df['scratchpad_used'] == True].to_excel(writer, sheet_name='Scratchpad Users', index=False)
    
    # Crashed runs
    df[df['crashed'] == True].to_excel(writer, sheet_name='Crashes', index=False)
    
    # Summary statistics
    summary_stats = df.groupby('params.model.name').agg({
        'crashed': 'mean',
        'total_steps': 'mean',
        'scratchpad_used': 'mean',
        'budget_burn_rate': 'mean'
    })
    summary_stats.to_excel(writer, sheet_name='By Model')
```

### Generate Report
```python
import datetime

report = f"""
VendoMini Experiment Report
Generated: {datetime.datetime.now()}

SUMMARY
=======
Total Runs: {len(df)}
Crash Rate: {df['crashed'].mean():.1%}
Scratchpad Usage: {df['scratchpad_used'].mean():.1%}
Bankruptcy Rate: {df['went_bankrupt'].mean():.1%}

BY MODEL
========
"""

for model in df['params.model.name'].unique():
    model_df = df[df['params.model.name'] == model]
    report += f"\n{model}:\n"
    report += f"  Runs: {len(model_df)}\n"
    report += f"  Crash Rate: {model_df['crashed'].mean():.1%}\n"
    report += f"  Mean Survival: {model_df['total_steps'].mean():.1f} steps\n"
    report += f"  Scratchpad Usage: {model_df['scratchpad_used'].mean():.1%}\n"

with open('results/experiment_report.txt', 'w') as f:
    f.write(report)

print("Report saved to results/experiment_report.txt")
```
