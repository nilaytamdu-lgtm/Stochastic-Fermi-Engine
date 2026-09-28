from src.engine import StochasticEngine, FermiComponent
from src.visualizer import plot_results

def test_v2():
    print("Initializing mock components v2...")
    components = [
        FermiComponent(name="Population", factor_type="continuous", p10=2.0e6, p90=3.0e6, deterministic_point=2.7e6, explanation="Chicago Pop", linked_factors=[]),
        FermiComponent(name="Economy Status", factor_type="continuous", p10=0.8, p90=1.2, deterministic_point=1.0, explanation="Macro multiplier", linked_factors=["Adoption Rate"]),
        FermiComponent(name="Adoption Rate", factor_type="continuous", p10=0.01, p90=0.10, deterministic_point=0.05, explanation="Scooter users", linked_factors=["Economy Status"]),
        FermiComponent(name="Regulatory Approval", factor_type="binary", p10=0, p90=1, deterministic_point=0.8, explanation="80% chance city approves", linked_factors=[])
    ]
    
    engine = StochasticEngine(n_samples=50000)
    print("Running simulation v2...")
    result = engine.run_simulation(components)
    
    print(f"Deterministic: {result.deterministic_total:,.0f}")
    print(f"True Mean: {result.true_mean:,.0f}")
    print(f"True Median: {result.true_median:,.0f}")
    
    print("Test v2 complete.")

if __name__ == "__main__":
    test_v2()
