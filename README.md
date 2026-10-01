# Strategies for preventing and reversing polarized online discourse

This repository contains code for the model explored in the paper *Strategies for preventing and reversing polarized online discourse* ([link to our preprint](https://doi.org/10.48550/arXiv.2606.18226)). The model is built using the [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) framework for agent-based modelling and is written in the Julia language.

## System requirements

Running the model requires a Julia environment (we used Julia 1.11 for simulations but 1.12 is also tested) with dependencies installed. Details on how to install and run Julia can be found at the official Julia [getting started page](https://docs.julialang.org/en/v1/manual/getting-started/) and dependencies used are listed in [`Project.toml`](Project.toml). Installing Julia and package dependencies should only take minutes.

## Quick-start example

From the repository root, activating the environment and installing all dependencies can be done with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Running the model with all parameters set to default values can then be done with (runs 10 replicates and takes about 5 minutes):

```bash
julia --project=. run_model.jl
```

To explore the parameter space, follow the edit instructions in [`run_model.jl`](run_model.jl).

## Repository structure

The code for the model is presented in the following files:

- [`mod_structs.jl`](model/mod_structs.jl) contains definitions of the basic data structures used to run the model.
- [`mod_init_functions.jl`](model/mod_init_functions.jl) contains functions for initializing the model.
- [`mod_step_functions.jl`](model/mod_step_functions.jl) contains code for the model dynamics.
- [`default_params.jl`](default_params.jl) contains default parameter values.

In addition there is code to run replicates of model simulations and display the outcome state:

- [`run_model.jl`](run_model.jl) contains code for running the model and space to change model parameters.
- [`data_aggregation_functions.jl`](data_aggregation_functions.jl) contains code for calculating the outcome states (Consensus, Constrained polarization, and Runaway polarization, as well as, latent extremism).

## Running simulations

The model uses the [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) framework for agent-based modelling and its functionality for recording simulations. Any simulation can be configured to return a table with the state of any of the fields in the model and its (`Individual`) agents at any selected simulation time points. Of main interest here is the `attitude` feature of each `Individual` which will be returned as a matrix (`N_attitude_dimensions` × `N_identity_dimensions`) for each `Individual` in the simulation. Running a single simulation of 5 million time steps under default parameter settings should take about a minute or less on most modern consumer hardware.

## Data aggregation

The results presented in our paper are all obtained by running many replicates over reported ranges of parameter values and aggregating and classifying outcomes. We provide code for classifying aggregated outcome states for a given parameter set in [`data_aggregation_functions.jl`](data_aggregation_functions.jl).
