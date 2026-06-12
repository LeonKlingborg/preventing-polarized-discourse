using Agents
using Distributions
using StatsBase
using Distances
using StaticArrays

# The object used to assemble a new expression in the model
struct PendingExpression
    author_ID::Int64
    attitude_mask::SVector{N_attitude_dimensions,Bool} # To make the model able to operate on multiple attitude dimensions at one time each PendingExpression carries a boolean mask for which dimensions are relevant for the expression
    exposed_agents_IDs::Vector{Int64} # The IDs of each agent exposed to the expression in question
    depth_level::Int64 # How many levels down in an interaction thread an expression sits
end

# The object used to store expressions memorised by agents
mutable struct MemorisedExpression
    attitude::SVector{N_attitude_dimensions,Float64}
    attitude_mask::SVector{N_attitude_dimensions,Bool}
    author_ID::Int64
    is_relevant::Bool # A boolean that is mutated to false or true each time an individual with the expression memorised looks at which memories are relevant for a given interaction (whether they share the same attitude dimension)
    ref_index::MVector{N_attitude_dimensions,Bool} # The overlap between MemorisedExpresseion.attitude_mask and PendingExpression.attitude_mask
end

# The object representing the agents in the model
@agent struct Individual(NoSpaceAgent)
    attitude::MMatrix{N_attitude_dimensions,N_identity_dimensions,Float64,N_attitude_dimensions*N_identity_dimensions} # The attitude matrix of the agent
    attitude_std::SMatrix{N_attitude_dimensions,N_identity_dimensions,Float64,N_attitude_dimensions*N_identity_dimensions} # The standard deviations of each attitude (matrix of 1s for all simulations)
    memorised_exposed::Vector{MemorisedExpression} # The expressions memorised by the agent that are from interactions where the agent was exposed to another agent's expression
    memorised_self::Vector{MemorisedExpression} # The self-expressions memorised by the agent, i.e. the expressions from interactions where the agent was the initiator
    N_interacted::Int64 # The number of interactions the agent has had, used to determine which memorised expression to overwrite when memorising a new one
    N_initiated::Int64 # The number of interactions the agent has initiated, used to determine which memorised expression to overwrite when memorising a new one
    ingroup::Vector{MVector{N_agents,Bool}} # Whether or not each other agent is considered an ingroup member, overwritten each time the agent expresses a new attitude
    identity_use::MMatrix{N_attitude_dimensions,N_identity_dimensions,Int64,N_attitude_dimensions*N_identity_dimensions} # The number of times the agent has expressed each of their attitudes (each cell in the attitude matrix)
end

mutable struct ModelProperties
    N_agents::Int64
    N_attitude_dimensions::Int64
    N_identity_dimensions::Int64
    N_memorised_other::Int64
    N_memorised_self::Int64
    N_search_points::Int64 # Generalises further than the paper
    inclusivity_scaler::Float64 # Generalises further than the paper
    attitude_shift_proportion::Float64 # The relative step size
    self_weight::Float64
    pull::Float64

    random_seed::Int64

    sigmoid_slope_mod::Float64 # The steepness of the sigmoid used to select the final expressed attitude
    sigmoid_slope_id::Float64 # The steepness of the sigmoid used to select salient identity dimension
    soft_overton_cost::Float64
    hard_overton_cost::Float64
    hard_overton_windows::Vector{Tuple{Float64,Float64}} # Capable of having different windows on different attitude dimensions (further than paper)

    dim_probs::SVector{N_attitude_dimensions,Float64} # The probability of selecting each attitude dimensions for attitude expression (sums to one)

    intervention::Symbol # Whether interventions are applied and which type of intervention [:no_intervention, :normal, :after]
    intervention_time::Int64 # If model.intervention == :normal, intervention_time is the model time at which interventions are applied, if == :after, intervention_time is the model intervention time after onset of polarization 

    cost_increment::Float64 # The size of the cost hike
    cost_hike_intervention::Bool
    soft_hike_intervention::Bool

    window_change_intervention::Bool
    window_change::Float64 # The width of the Overton window after intervention

    elite_anchor_intervention::Bool
    elite_anchor_strength::Float64 # The weight of the moderate elite term in the utility function

    equal_dimprob_intervention::Bool

    init_agent_attitude_dist::Distribution # The distribution from which initial attitudes are sampled
    init_agent_attitude_std::Float64

    track_polarization::Bool # Whether to track if the model population has yet reached polarization during simulation
    only_esc::Bool # Whether tracked polarization only refers to runaway polarization
    
    # Below are placeholder objects to avoid memory allocation, not to be used as parameters 
    pending_expressions::Vector{PendingExpression}
    reference_points::Vector{SVector{N_attitude_dimensions,Float64}}
    in_ag_per_id::Vector{MVector{N_agents,Bool}}
    out_ag_per_id::Vector{MVector{N_agents,Bool}}
    overton_ptuple::Tuple{MVector{N_attitude_dimensions,Float64},MVector{N_attitude_dimensions,Float64}}
    freq_vec::MVector{N_agents,Float64}
    distance_vec::MVector{N_agents,Float64}

    no_in_context::Int64
    no_out_context::Int64
    
    polarized_time::Int64
    polarized_bool::Bool
end

mutable struct ModelParams
    N_agents::Int64
    N_attitude_dimensions::Int64
    N_identity_dimensions::Int64
    N_memorised_other::Int64
    N_memorised_self::Int64
    init_agent_attitude_overton_proportion::Float64
    init_agent_attitude_std::Float64
    N_search_points::Int64
    inclusivity_scaler::Float64
    attitude_shift_proportion::Float64
    self_weight::Float64 
    random_seed::Int64
    pull::Float64
    sigmoid_slope_mod::Float64
    sigmoid_slope_id::Float64
    soft_overton_cost::Float64
    hard_overton_cost::Float64
    hard_overton_scalers::Vector{Float64}
    dim_probs::Vector{Float64}
    intervention::Symbol
    intervention_time::Int64
    cost_increment::Float64
    cost_hike_intervention::Bool
    soft_hike_intervention::Bool
    window_change_intervention::Bool
    window_change::Float64
    elite_anchor_intervention::Bool
    elite_anchor_strength::Float64
    equal_dimprob_intervention::Bool
    track_polarization::Bool
    only_esc::Bool
end

# Function to help create model objects for different values of model parameters
function model_spec_rec(td; noex_params=[], param_dicts=[], inner_use = false)
    ks = keys(td)
    ks = [k for k in ks if ((td[k] isa AbstractVector))]
    exp_ks = [k for k in ks if ((length(td[k]) > 1) & (k ∉ noex_params))]
    # Moves value with key :random_seed to last index in exp_ks
    if :random_seed in exp_ks
        deleteat!(exp_ks, findfirst(isequal(:random_seed), exp_ks))
        push!(exp_ks, :random_seed)
    end
    group_vars = [v for v in exp_ks if v != :random_seed]
    if length(exp_ks) == 0
        if inner_use
            return [td]
        else
            return [td], group_vars
        end
    else
        fix_k = first(exp_ks)
        for fix_v in td[fix_k]
            fix_td = copy(td)
            fix_td[fix_k] = [fix_v]
            param_dicts = [param_dicts ; model_spec_rec(fix_td; noex_params=noex_params, inner_use=true)]
        end
        if inner_use
            return param_dicts
        else
            return param_dicts, group_vars
        end
    end
end

# Function to help create model objects for different values of model parameters
function get_param_dicts(model_params)
    scan_param_set, group_vars = model_spec_rec(model_params)
    for ps in scan_param_set
        for k in keys(ps)
            if ((ps[k] isa AbstractVector))
                if length(ps[k]) == 1
                    ps[k] = ps[k][1]
                end
            end
        end
    end
    return scan_param_set, group_vars
end

