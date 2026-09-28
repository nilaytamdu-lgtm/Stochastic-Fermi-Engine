from src.engine import StochasticEngine, FermiComponent
from src.visualizer import plot_results

def test_engine():
    print("Initializing mock components...")
    # Mocking Chicago Scooter rides
    components = [
        FermiComponent(name="Population", p10=2.0e6, p90=3.0e6, deterministic_point=2.7e6, explanation="Chicago Pop"),
        FermiComponent(name="Adoption Rate", p10=0.01, p90=0.10, deterministic_point=0.05, explanation="Scooter users"),
        FermiComponent(name="Rides per user", p10=20, p90=200, deterministic_point=100, explanation="Annual rides"),
    ]
    
    engine = StochasticEngine(n_samples=50000)
    print("Running simulation...")
    result = engine.run_simulation(components)
    
    print(f"Deterministic: {result.deterministic_total:,.0f}")
    print(f"True Mean: {result.true_mean:,.0f}")
    print(f"True Median: {result.true_median:,.0f}")
    
    print("Generating plot...")
    plot_results(result, output_path="test_plot.png", title="Mock Market Size")
    print("Test complete.")

if __name__ == "__main__":
    test_engine()
