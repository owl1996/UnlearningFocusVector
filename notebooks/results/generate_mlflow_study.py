import nbformat as nbf
import os

# Create a new Jupyter notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Title and Imports
cells.append(nbf.v4.new_markdown_cell("# MLFlow Results Study\nAn analysis of the unlearning performance tracked by MLFlow runs, mapping `rUA`, `FID`, and running times over different configurations."))

code_imports = """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10
})
"""
cells.append(nbf.v4.new_code_cell(code_imports))

# Load MLFlow Data
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loader & Preprocessing\nLoad the aggregated `mlruns_parsed.csv` tracked automatically during unlearning executions."))

code_data = """
csv_path = '../../mlruns_parsed.csv'

try:
    df = pd.read_csv(csv_path)
    if 'seed' in df.columns:
        df.drop(columns=['seed'], inplace=True)

    # Standardize Column Names
    rcolumns = {
        'rUA (%)' : 'rUA',
        'UA (%)' : 'UA',
        'RA (%)' : 'RA',
        'TA (%)' : 'TA'
    }
    df = df.rename(columns=rcolumns)

    # Standardize Method Names if applicable based on the original template
    rmethods = {'NGPlus - GradFocus' : 'NGPlus - F',
                'NGPlus - GradMask' : 'NGPlus - PROB',
                'NGPlus - ANDMask' : 'NGPlus - AND',
                'SRL - GradFocus' : 'SRL - F',
                'SRL - GradMask' : 'SRL - PROB',
                'SRL - ANDMask' : 'SRL - AND',
                'SalUn' : 'SRL - SalUn',
                'SCRUB - GradFocus' : 'SCRUB - F',
                'SCRUB - GradMask' : 'SCRUB - PROB',
                'SCRUB - ANDMask' : 'SCRUB - AND',
                'OS_unlearn' : 'OS_Unlearn'
                }

    for key, value in rmethods.items():
        df.loc[df['Methods'] == key, 'Methods'] = value
        
    # Example cumulative sum transformation if studying RTE progressively filtering per architecture
    grp_cols = ['Methods', 'num_indexes_to_replace', 'class_to_replace', "arch", "dataset"]
    if 'RTE' in df.columns:
        df['RTE_cum'] = df.groupby(grp_cols)['RTE'].cumsum(axis=0)

    print("Data loaded successfully.")
    display(df.head())
except Exception as e:
    print(f"Failed to process CSV data: {e}")
"""
cells.append(nbf.v4.new_code_cell(code_data))


# Box Plot Analysis
cells.append(nbf.v4.new_markdown_cell("## 2. Performance Comparison\nCompare relative Unlearning Accuracy (rUA) and Retained Accuracy (FID) across implemented methods."))

code_box = """
# Filter data for specific replacement index if needed (e.g., 2000 instances)
df_filtered = df[df['class_to_replace'] == -1].copy()

if not df_filtered.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # rUA Plot
    sns.boxplot(data=df_filtered, x='Methods', y='rUA', ax=axes[0], palette='Set2')
    axes[0].set_title('Relative Unlearning Accuracy (rUA) per Method')
    axes[0].set_ylabel('rUA (%)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # FID Plot
    sns.boxplot(data=df_filtered, x='Methods', y='FID', ax=axes[1], palette='Set3')
    axes[1].set_title('Retained Accuracy (FID) per Method')
    axes[1].set_ylabel('FID (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
else:
    print("No evaluation data matched the filter.")
"""
cells.append(nbf.v4.new_code_cell(code_box))


# Scatter Epoch vs RTE vs rUA
cells.append(nbf.v4.new_markdown_cell("## 3. Training Cost Analysis\nObserve the tradeoff between Unlearning Epochs, Time Spent (RTE), and Performance (rUA)."))

code_scatter = r"""
if not df_filtered.empty and 'Unlearn epochs' in df_filtered.columns and 'RTE' in df_filtered.columns:
    # NeurIPS Aesthetics
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    plt.figure(figsize=(10, 7))
    
    # Base Scatter Plot
    scatter = sns.scatterplot(
        data=df_filtered, 
        x='RTE', 
        y='rUA', 
        hue='Methods',
        style='Methods',
        s=150,           # Static larger size for visibility
        alpha=0.85,
        edgecolor='k',   # Dark edge for contrast
        palette='tab10'
    )
    
    # ---------------------------------------------
    # Pareto Frontier Calculation (Minimize RTE, Maximize rUA)
    # ---------------------------------------------
    # Sort by RTE ascending, then rUA descending
    df_sorted = df_filtered.sort_values(by=['RTE', 'rUA'], ascending=[True, False])
    
    pareto_front = []
    max_rUA_so_far = -float('inf')
    
    for index, row in df_sorted.iterrows():
        if row['rUA'] > max_rUA_so_far:
            pareto_front.append((row['RTE'], row['rUA']))
            max_rUA_so_far = row['rUA']
            
    if pareto_front:
        pareto_x, pareto_y = zip(*pareto_front)
        # Plot the Pareto envelope
        plt.plot(pareto_x, pareto_y, '--', color='red', linewidth=2.5, alpha=0.7, label='Pareto Frontier')

    plt.title('Performance vs Computational Cost\\nPareto Frontier Analysis')
    plt.xlabel('Run Time Elapsed (s) $\\downarrow$')
    plt.ylabel('Relative Unlearning Accuracy (rUA %) $\\uparrow$')
    
    # Move legend outside elegantly
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("pareto_analysis_neurips.pdf", format='pdf', bbox_inches='tight')
    plt.show()
else:
    print("Required columns for scatter plot are missing.")
"""
cells.append(nbf.v4.new_code_cell(code_scatter))

# Assign cells to notebook and save
nb.cells = cells
output_file = 'notebooks/results/mlflow_results_study.ipynb'

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    nbf.write(nb, f)

print(f"Jupyter notebook '{output_file}' created successfully.")
