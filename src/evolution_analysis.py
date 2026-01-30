
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.animation as animation
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
ERA_MAP = {
    "Bach": 1,
    "Mozart": 2,
    "Chopin": 3,
    "Debussy": 4
}
ERA_NAMES = {
    1: "Baroque (Bach)",
    2: "Classical (Mozart)",
    3: "Romantic (Chopin)",
    4: "Impressionist (Debussy)"
}
COLORS = {
    "Bach": "#636efa",
    "Mozart": "#ab63fa",
    "Chopin": "#EF553B",
    "Debussy": "#00cc96"
}

def load_and_merge_data():
    """Loads harmonic, melodic, and rhythmic features and merges them."""
    try:
        harmonic = pd.read_csv(DATA_DIR / "features/harmonic_features.csv")
        melodic = pd.read_csv(DATA_DIR / "features/melodic_features.csv")
        rhythmic = pd.read_csv(DATA_DIR / "features/rhythmic_features.csv")
        
        # Merge on filepath (or title/composer, but filepath is safest unique ID)
        # Assuming mxl_path is consistent
        merged = harmonic.merge(melodic[['mxl_path', 'pitch_range_semitones', 'avg_melodic_interval']], on='mxl_path', suffixes=('', '_mel'))
        merged = merged.merge(rhythmic[['mxl_path', 'rhythmic_pattern_entropy', 'std_note_duration']], on='mxl_path', suffixes=('', '_rhy'))
        
        # Add ordinal era
        merged['era_ordinal'] = merged['composer_label'].map(ERA_MAP)
        
        return merged
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_velocities(df):
    """Computes 'velocity' (mean differences) and 'acceleration' (diff of diffs)."""
    results = []
    
    for feature in FEATURES_TO_ANALYZE:
        means = df.groupby('composer_label')[feature].mean()
        
        # Velocities (First Differences)
        v_class = means['Mozart'] - means['Bach']      # Baroque -> Classical
        v_rom = means['Chopin'] - means['Mozart']      # Classical -> Romantic
        v_imp = means['Debussy'] - means['Chopin']     # Romantic -> Impressionist
        
        # Accelerations (Difference in Differences)
        a_rom = v_rom - v_class  # Did change accelerate entering Romantic era?
        a_imp = v_imp - v_rom    # Did change accelerate entering Impressionism?
        
        # T-tests for Velocities (Independent samples)
        # Function to get p-value for diff between two groups
        def get_pval(g1, g2):
            return stats.ttest_ind(
                df[df['composer_label'] == g1][feature],
                df[df['composer_label'] == g2][feature],
                equal_var=False
            ).pvalue

        p_class = get_pval('Bach', 'Mozart')
        p_rom = get_pval('Mozart', 'Chopin')
        p_imp = get_pval('Chopin', 'Debussy')

        results.append({
            'feature': feature,
            'v_classical': v_class,
            'p_classical': p_class,
            'v_romantic': v_rom,
            'p_romantic': p_rom,
            'v_impressionist': v_imp,
            'p_impressionist': p_imp,
            'accel_romantic': a_rom,
            'accel_impressionist': a_imp
        })
        
    return pd.DataFrame(results)

def create_evolution_animation(df):
    """Generates an animated GIF of feature distributions evolving."""
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Global limits for each feature to keep frames consistent
    limits = {}
    for i, feature in enumerate(FEATURES_TO_ANALYZE):
        min_val = df[feature].min()
        max_val = df[feature].max()
        padding = (max_val - min_val) * 0.1
        limits[feature] = (min_val - padding, max_val + padding)

    def update(frame):
        # Frame 0-3 correspond to the 4 composers accumulating
        # We cap index at 3 to avoid out of bounds in the 'pause' frames if we had them
        current_step = min(frame, 3)
        
        for i, feature in enumerate(FEATURES_TO_ANALYZE):
            ax = axes[i]
            ax.clear()
            ax.set_title(feature.replace('_', ' ').title(), fontsize=12)
            ax.set_xlim(limits[feature])
            ax.set_yticks([]) # Hide density counts for cleaner look
            
            # Plot active composers up to current frame
            for idx in range(current_step + 1):
                composer = COMPOSER_ORDER[idx]
                subset = df[df['composer_label'] == composer]
                if len(subset) > 1: # KDE needs > 1 point
                    sns.kdeplot(
                        subset[feature], 
                        ax=ax, 
                        color=COLORS[composer], 
                        fill=True, 
                        alpha=0.6, 
                        label=composer if i==0 else None,
                        linewidth=2
                    )
            
            # Add arrows/text for the newest transition
            if current_step > 0:
                prev_comp = COMPOSER_ORDER[current_step - 1]
                curr_comp = COMPOSER_ORDER[current_step]
                
                prev_mean = df[df['composer_label'] == prev_comp][feature].mean()
                curr_mean = df[df['composer_label'] == curr_comp][feature].mean()
                
                # Draw arrow
                # Y position varies slightly to avoid overlap
                y_pos = ax.get_ylim()[1] * 0.1 * current_step + 0.05
                ax.annotate(
                    "", 
                    xy=(curr_mean, y_pos), 
                    xytext=(prev_mean, y_pos),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5)
                )

        fig.suptitle(f"Era: {ERA_NAMES[current_step + 1]}", fontsize=16, fontweight='bold')
        if current_step == 3:
            fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # Create animation
    # Frames: 0=Bach, 1=+Mozart, 2=+Chopin, 3=+Debussy
    anim = animation.FuncAnimation(fig, update, frames=4, interval=1500)
    
    output_path = FIGURE_DIR / "evolution_timelapse.gif"
    try:
        anim.save(output_path, writer='pillow', fps=1)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Could not save animation (requires ffmpeg or pillow): {e}")

def main():
    print("Loading data...")
    df = load_and_merge_data()
    if df is None:
        return

    print("Analyzing evolutionary velocities...")
    stats_df = analyze_velocities(df)
    stats_path = DATA_DIR / "stats/evolution_coefficients.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved to {stats_path}")
    print("\nEvolution Summary:")
    print(stats_df.round(3).to_string())

    print("\nGenerating animation...")
    create_evolution_animation(df)

if __name__ == "__main__":
    main()
