# Stochastic Fermi-Poisson Engine

A neuro-symbolic framework for probabilistic market sizing and strategic estimation.

## Overview
Traditional corporate market sizing (Total Addressable Market calculations) relies on deterministic point estimates. When multiplied together, these point estimates systematically commit the "Flaw of Averages," hiding compounding variance and severe downside tail risks. 

This repository contains a two-stage computational pipeline that solves this issue by combining Large Language Models (LLMs) with advanced stochastic modeling:
1. **Stage 1 (Elicitation):** An LLM agent extracts the dependency tree of a market sizing problem and estimates the pessimistic (p10) and optimistic (p90) bounds.
2. **Stage 2 (Stochastic Engine):** A Python-based Monte Carlo engine converts these percentiles into log-normal parameters and executes a 10,000-draw simulation. It utilizes a Gaussian Copula to account for variable covariance and routes variables dynamically based on continuous vs. binary types.

## File Structure
* `src/elicitor.py`: Gemini AI integration enforcing Pydantic structured JSON output for parameter boundaries.
* `src/engine.py`: The core Monte Carlo and Poisson-Lognormal mathematical engine.
* `src/visualizer.py`: Generates the probability density histograms comparing the deterministic average against the true stochastic median.

## Academic Research
This repository serves as the computational proof-of-concept for an ongoing research project regarding Artificial Intelligence, decision theory, and the Flaw of Averages in strategic forecasting.
