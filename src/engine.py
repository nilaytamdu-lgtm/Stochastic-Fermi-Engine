import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class FermiComponent:
    name: str
    factor_type: str
    p10: float
    p90: float
    deterministic_point: float
    explanation: str
    linked_factors: List[str] = field(default_factory=list)

@dataclass
class SimulationResult:
    lambda_samples: np.ndarray
    poisson_samples: np.ndarray
    deterministic_total: float
    true_median: float
    true_mean: float
    true_std: float
    p05: float
    p95: float
    probability_of_ruin: float
    
class StochasticEngine:
    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples
        self.z_score = 1.28155 # Z-score for 10th to 90th percentile
        self.calibration_penalty = 1.15 # Widens variance to fix LLM overconfidence

    def run_simulation(self, components: List[FermiComponent]) -> SimulationResult:
        N = len(components)
        deterministic_total = 1.0
        for comp in components:
            deterministic_total *= comp.deterministic_point
            
        # 1. Build Correlation Matrix from LLM text links
        R = np.eye(N)
        name_to_idx = {c.name: i for i, c in enumerate(components)}
        
        for i, comp in enumerate(components):
            for link in comp.linked_factors:
                if link in name_to_idx:
                    j = name_to_idx[link]
                    if i != j:
                        R[i, j] = 0.6 # Default positive correlation
                        R[j, i] = 0.6
                        
        # Ensure Positive Semi-Definite for Copula
        vals, vecs = np.linalg.eigh(R)
        vals[vals < 1e-8] = 1e-8
        R = vecs @ np.diag(vals) @ vecs.T
        
        # 2. Generate Gaussian Copula Z-scores
        Z = np.random.multivariate_normal(np.zeros(N), R, size=self.n_samples)
        
        # 3. Convert Z-scores to Uniform Percentiles
        U = stats.norm.cdf(Z)
        
        # 4. Map Percentiles to specific statistical distributions
        samples = np.zeros((self.n_samples, N))
        
        for i, comp in enumerate(components):
            if comp.factor_type.lower() == 'binary':
                # Binary Event (Bernoulli)
                # If U < probability, success (1.0), else failure (0.0)
                samples[:, i] = np.where(U[:, i] < comp.deterministic_point, 1.0, 0.0)
            else:
                # Continuous Multiplicative (Log-Normal)
                ln_p10 = np.log(comp.p10)
                ln_p90 = np.log(comp.p90)
                mu = (ln_p90 + ln_p10) / 2.0
                # Apply Calibration Penalty to Standard Deviation
                sigma = ((ln_p90 - ln_p10) / (2.0 * self.z_score)) * self.calibration_penalty
                
                samples[:, i] = stats.lognorm.ppf(U[:, i], s=sigma, scale=np.exp(mu))
                
        # 5. Compound total and run through Poisson point process
        lambda_samples = np.prod(samples, axis=1)
        poisson_samples = np.random.poisson(lambda_samples)
        
        # Calculate key metrics
        true_median = float(np.median(poisson_samples))
        true_mean = float(np.mean(poisson_samples))
        true_std = float(np.std(poisson_samples))
        p05 = float(np.percentile(poisson_samples, 5))
        p95 = float(np.percentile(poisson_samples, 95))
        
        threshold = deterministic_total * 0.5
        prob_ruin = float(np.mean(poisson_samples < threshold))
        
        return SimulationResult(
            lambda_samples=lambda_samples,
            poisson_samples=poisson_samples,
            deterministic_total=deterministic_total,
            true_median=true_median,
            true_mean=true_mean,
            true_std=true_std,
            p05=p05,
            p95=p95,
            probability_of_ruin=prob_ruin
        )
