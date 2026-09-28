import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .engine import SimulationResult

def plot_results(result: SimulationResult, output_path: str = "distribution_plot.png", title: str = "Market Size Distribution"):
    plt.figure(figsize=(12, 7))
    
    # Filter extreme outliers for visualization clarity (e.g., top 1%)
    cap = np.percentile(result.poisson_samples, 99)
    plot_data = result.poisson_samples[result.poisson_samples < cap]
    
    sns.histplot(plot_data, bins=50, kde=True, color='steelblue', stat='density', alpha=0.6)
    
    # Vertical lines for key metrics
    plt.axvline(result.deterministic_total, color='red', linestyle='--', linewidth=2, label=f'Deterministic Point: {result.deterministic_total:,.0f}')
    plt.axvline(result.true_median, color='green', linestyle='-', linewidth=2, label=f'True Median: {result.true_median:,.0f}')
    plt.axvline(result.true_mean, color='orange', linestyle='-', linewidth=2, label=f'True Mean: {result.true_mean:,.0f}')
    
    # 90% Confidence Interval Shading
    plt.axvspan(result.p05, result.p95, color='gray', alpha=0.15, label=f'90% Interval: {result.p05:,.0f} to {result.p95:,.0f}')
    
    plt.title(title + "\n(Poisson-Lognormal Compound Distribution)", fontsize=14, fontweight='bold')
    plt.xlabel("Estimated Count / Volume", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")
    plt.close()
