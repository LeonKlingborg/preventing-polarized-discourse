using DataFrames
using ProgressBars
using Random
using Distributions
using Arrow
using StaticArrays

include("./default_params.jl")

# Core parameters that control dimensionality in different ways are specfied as constants to help with making execution more efficient
const N_attitude_dimensions = model_params[:N_attitude_dimensions]
const N_agents = model_params[:N_agents]
const N_identity_dimensions = model_params[:N_identity_dimensions]
const N_search_points = model_params[:N_search_points]

include("./model/mod_structs.jl")
include("./model/mod_init_functions.jl")
include("./model/mod_step_functions.jl")

n = 10_000_000 # The number of simulated expressed attitudes
n_data_points = 500 # How many times the model state is recorded over n
when_interval = n ÷ n_data_points

### Change parameter values here

# model_params[:pull] = 0.5

###


model_params[:random_seed] = 2389

model = dict2model(model_params)

adata = [:attitude, :identity_use, :ingroup] #, :util, :in_mean, :out_mean, :udiff, :ingroup]
mdata = collect(keys(model_params))
mdata = [nv == :hard_overton_scalers ? :hard_overton_windows : nv for nv in mdata]
mdata = [nv == :init_agent_attitude_overton_proportion ? :init_agent_attitude_dist : nv for nv in mdata]


adf, mdf = run!(model, n;
    adata=adata, 
    mdata=mdata,
    obtainer=deepcopy,
    when=collect(when_interval:when_interval:n),
    init=true,
    showprogress=true
)