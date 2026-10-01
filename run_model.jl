include("./default_params.jl")

####################################
# Model configuration for testing  #
####################################


steps_per_run = 5_000_000   # The number of steps in model simulations (corresponds to the number of attitude updates).
number_of_replicates = 10   # The number of samples to run for the given parameter set.
first_random_seed = 1001

window_width = 5.0          # The width of the hard Overton window.
attention_bias = 0.0        # The attention bias between topics (attitude dimensions).
# As is described in the paper, attention bias of 0.0 corresponds to equal attention to both attitude dimensions,
# and attention bias of 1.0 corresponds to exclusive attention to the first attitude dimension.

# Edit further model parameters by giving values below.
parameter_overrides = Dict{Symbol, Any}(
    # :pull => 0.5,
    # :N_agents => 200,
)








########################################


for parameter in keys(parameter_overrides)
    haskey(model_params, parameter) ||
        error("Unknown model parameter: $parameter")
end
merge!(model_params, parameter_overrides)

# A few (non-exhaustive) tests to ensure that the parameters are valid.
steps_per_run >= 500_000 ||
    error("steps_per_run must be greater than 500,000 to allow for reasonable assessment of latent extremism.")
model_params[:N_attitude_dimensions] == 2 ||
    error("run_model.jl currently supports exactly two attitude dimensions.")
0.0 <= attention_bias <= 1.0 ||
    error("attention_bias must be between 0 and 1.")
0.0 <= model_params[:pull] <= 1.0 ||
    error("pull must be between 0 and 1.")

# Loading required packages
using DataFrames
using ProgressBars
using Random
using Distributions
using Arrow
using StaticArrays

# Core parameters that control dimensionality are specfied as constants to make execution more efficient.
const N_attitude_dimensions = model_params[:N_attitude_dimensions]
const N_agents = model_params[:N_agents]
const N_identity_dimensions = model_params[:N_identity_dimensions]
const N_search_points = model_params[:N_search_points]

# Pulling in the code required to run the model and classify outcome states.
include("./model/mod_structs.jl")
include("./model/mod_init_functions.jl")
include("./model/mod_step_functions.jl")
include("./data_aggregation_functions.jl")

# The hard Overton window is specified by supplying a vector with one value per attitude dimension specifying the window width. 
model_params[:hard_overton_scalers] = [window_width for _ in 1:model_params[:N_attitude_dimensions]]

# The attention bias is specified by supplying the probability of selecting each attitude dimension. The numbers in the vector therefore need to sum to 1.
p = (attention_bias + 1.0) / 2.0
model_params[:dim_probs] = [p, 1-p]


# Helper function to create a model from each random seed provided.
seeds = collect(first_random_seed:(first_random_seed + number_of_replicates - 1))
function model_from_seed(seed)
    params = copy(model_params)
    params[:random_seed] = Int(seed)
    return dict2model(params)
end

adata = [:attitude, :identity_use]  # Which agent-level data to record
mdata = collect(keys(model_params)) # Collect all model parameters to record
mdata = [nv == :hard_overton_scalers ? :hard_overton_windows : nv for nv in mdata]
mdata = [nv == :init_agent_attitude_overton_proportion ? :init_agent_attitude_dist : nv for nv in mdata]


println("""

Running model:
  steps per replicate: $steps_per_run
  replicates:          $number_of_replicates
  window width:        $window_width
  attention bias:      $attention_bias
""")


# Running model replicates
adf, mdf, final_models = ensemblerun!(
    model_from_seed,
    steps_per_run;
    seeds=seeds,
    adata=adata,
    mdata=mdata,
    obtainer=deepcopy,
    when=[steps_per_run-200_000, steps_per_run], # The model times at which states are recorded. We record the final state and an earlier state to be able to assess latent extremism.
    when_model=[],
    init=true,
    showprogress=true,
)


# Saving the raw simulation outputs
agent_path = joinpath(@__DIR__, "data_agent_attitudes.arrow")
model_path = joinpath(@__DIR__, "data_model_parameters.arrow")
Arrow.write(agent_path, adf)
Arrow.write(model_path, mdf)

println("""

Aggregating data and classifying outcome states.
""")

# Classifying the outcome state of the given parameter set and the amount of latent extremism
agg_dict = data_prep(agent_path, model_path, assume_2_dims=true)

# Printing the outcome
println(
    "\nOutcome:\t $(agg_dict[:outcome_state])\n",
    "\nBreakdown:\t",
    " | consensus: $(round(agg_dict[:consensus_prop]; digits=2))",
    " | runaway: $(round(agg_dict[:runaway_prop]; digits=2))",
    " | contained: $(round(agg_dict[:contained_prop]; digits=2))\n",
)