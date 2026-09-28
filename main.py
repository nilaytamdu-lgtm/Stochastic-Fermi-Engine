import argparse
import os
from dotenv import load_dotenv
from src.elicitor import ElicitorAgent
from src.engine import StochasticEngine
from src.visualizer import plot_results

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Stochastic Fermi-Poisson Estimation")
    parser.add_argument("--query", type=str, required=True, help="The market sizing question (e.g., 'Annual electric scooter rides in Chicago')")
    parser.add_argument("--output", type=str, default="market_size_distribution.png", help="Output plot filename")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not set. Please add it to a .env file.")
        return

    print(f"\n--- Extracting Fermi Tree for: '{args.query}' ---")
    agent = ElicitorAgent()
    components = agent.elicit(args.query)
    
    print("\n[Fermi Tree Extracted]")
    for comp in components:
        print(f"- {comp.name}: Det={comp.deterministic_point:,.2f} | P10={comp.p10:,.2f} | P90={comp.p90:,.2f}")
        print(f"  Reasoning: {comp.explanation}")
        
    print("\n--- Running Stochastic Engine ---")
    engine = StochasticEngine()
    result = engine.run_simulation(components)
    
    print(f"\n[Simulation Results]")
    print(f"Deterministic Point Estimate: {result.deterministic_total:,.2f}")
    print(f"True Expected Mean:           {result.true_mean:,.2f}")
    print(f"True Median:                  {result.true_median:,.2f}")
    print(f"Standard Deviation:           {result.true_std:,.2f}")
    print(f"90% Confidence Interval:      [{result.p05:,.2f}, {result.p95:,.2f}]")
    print(f"Prob. of Ruin (< 50% of Det): {result.probability_of_ruin*100:.1f}%")
    
    print("\n--- Generating Visualization ---")
    plot_results(result, output_path=args.output, title=f"Distribution for: {args.query}")

if __name__ == "__main__":
    main()
