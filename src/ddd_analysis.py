
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
FIGURE_DIR = Path("figures/evolution")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_TO_ANALYZE = [
    "pitch_range_semitones",
    "dissonance_ratio",
    "harmonic_density_mean",
    "rhythmic_pattern_entropy"
]

COMPOSER_ORDER = ["Bach", "Mozart", "Chopin", "Debussy"]
ERA_MAP = {"Bach": 1, "Mozart": 2, "Chopin": 3, "Debussy": 4}

def load_and_merge_data():
    """Loads and prepares data with inferred mode."""
    try:
        harmonic = pd.read_csv(DATA_DIR / "features/harmonic_features.csv")
        melodic = pd.read_csv(DATA_DIR / "features/melodic_features.csv")
        rhythmic = pd.read_csv(DATA_DIR / "features/rhythmic_features.csv")
        
        merged = harmonic.merge(melodic[['mxl_path', 'pitch_range_semitones']], on='mxl_path')
        merged = merged.merge(rhythmic[['mxl_path', 'rhythmic_pattern_entropy']], on='mxl_path')
        
        # Infer Mode
        merged['inferred_mode'] = np.where(
            merged['chord_quality_major_pct'] > merged['chord_quality_minor_pct'],
            'Major',
            'Minor'
        )
        
        return merged
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_accelerations(df):
    """Calculates velocities and acceleration for a given dataframe subset."""
    results = {}
    
    for feature in FEATURES_TO_ANALYZE:
        means = df.groupby('composer_label')[feature].mean()
        
        # Ensure all composers exist in subset (handle edge case where a composer has 0 minor pieces)
        if len(means) < 4:
            continue
            
        v_class = means.get('Mozart', 0) - means.get('Bach', 0)
        v_rom = means.get('Chopin', 0) - means.get('Mozart', 0)
        v_imp = means.get('Debussy', 0) - means.get('Chopin', 0)
        
        accel_rom = v_rom - v_class
        accel_imp = v_imp - v_rom
        
        results[feature] = {
            'v_classical': v_class,
            'v_romantic': v_rom,
            'accel_romantic': accel_rom,
            'accel_imp': accel_imp
        }
    return results

def run_ddd_analysis(df):
    major_df = df[df['inferred_mode'] == 'Major']
    minor_df = df[df['inferred_mode'] == 'Minor']
    
    major_stats = calculate_accelerations(major_df)
    minor_stats = calculate_accelerations(minor_df)
    
    ddd_results = []
    
    print(f"Sample Sizes: Major (n={len(major_df)}), Minor (n={len(minor_df)})")
    
    for feature in FEATURES_TO_ANALYZE:
        if feature not in major_stats or feature not in minor_stats:
            continue
            
        maj = major_stats[feature]
        min_ = minor_stats[feature]
        
        # DDD: Did Minor enhance the Romantic acceleration more than Major?
        ddd_romantic = min_['accel_romantic'] - maj['accel_romantic']
        
        ddd_results.append({
            'feature': feature,
            'accel_rom_major': maj['accel_romantic'],
            'accel_rom_minor': min_['accel_romantic'],
            'DDD_romantic': ddd_romantic
        })
        
    return pd.DataFrame(ddd_results)

def plot_ddd(major_df, minor_df):
    """Visualizes the trajectories of Major vs Minor."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES_TO_ANALYZE):
        ax = axes[i]
        
        # Get means by composer & mode
        maj_means = major_df.groupby('composer_label')[feature].mean().reindex(COMPOSER_ORDER)
        min_means = minor_df.groupby('composer_label')[feature].mean().reindex(COMPOSER_ORDER)
        
        x_vals = range(4)
        
        ax.plot(x_vals, maj_means, marker='o', label='Major', color='#F4A261', linewidth=2.5)
        ax.plot(x_vals, min_means, marker='o', label='Minor', color='#264653', linewidth=2.5, linestyle='--')
        
        ax.set_xticks(x_vals)
        ax.set_xticklabels(COMPOSER_ORDER)
        ax.set_title(feature.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ddd_comparison.png")
    print(f"Plot saved to {FIGURE_DIR / 'ddd_comparison.png'}")

def main():
    df = load_and_merge_data()
    if df is None: return
    
    print("Running DDD (Major vs Minor) Analysis...")
    results = run_ddd_analysis(df)
    
    print("\nDDD Results (Romantic Acceleration Impact):")
    print(results.round(3).to_string())
    
    plot_ddd(df[df['inferred_mode'] == 'Major'], df[df['inferred_mode'] == 'Minor'])

if __name__ == "__main__":
    main()
