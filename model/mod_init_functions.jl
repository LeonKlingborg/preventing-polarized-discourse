using Random: Xoshiro
using Distributions
using Agents
using StaticArrays

function initialise_agent!(model::StandardABM, ag_ID::Int64)
    attitude = @MMatrix rand(abmrng(model), model.init_agent_attitude_dist, model.N_attitude_dimensions, model.N_identity_dimensions)
    attitude_std = @SMatrix fill(model.init_agent_attitude_std, model.N_attitude_dimensions, model.N_identity_dimensions)
    new_ag = Individual(
        ag_ID,
        attitude,
        attitude_std,
        MemorisedExpression[],
        MemorisedExpression[],
        0,
        0,
        [(@MVector ones(Float64, model.N_agents)) for _ in 1:model.N_identity_dimensions],
        @MMatrix zeros(Int64, model.N_attitude_dimensions, model.N_identity_dimensions)
    )
    add_agent!(new_ag, model)
end

function initialise_memories!(model::StandardABM)
    mask = @SVector ones(Bool, model.N_attitude_dimensions)
    for ag in allagents(model)
        ag.N_interacted = model.N_memorised_other
        ag.N_initiated = model.N_memorised_self
        auth_ids = sample(abmrng(model), [ag_id for ag_id in allids(model) if ag_id != ag.id], model.N_memorised_other, replace=true)
        for aid in auth_ids
            ag_auth = model[aid]
            op_dist = sample(abmrng(model), 1:model.N_identity_dimensions)
            att_mean = ag_auth.attitude[:,op_dist]
            att_std = ag_auth.attitude_std[:,op_dist]
            att = rand.(abmrng(model), Normal.(att_mean, att_std))
            author_ID = aid
            ref_index = @SVector ones(Bool, model.N_attitude_dimensions)
            mem_exp = MemorisedExpression(
                att, 
                mask, 
                author_ID, 
                true, 
                ref_index,
            )
            push!(ag.memorised_exposed, mem_exp)
        end
        for _ in 1:model.N_memorised_self
            op_dist = sample(abmrng(model), 1:model.N_identity_dimensions)
            att_mean = ag.attitude[:,op_dist]
            att_std = ag.attitude_std[:,op_dist]
            att = rand.(abmrng(model), Normal.(att_mean, att_std))
            author_ID = ag.id
            ref_index = @SVector ones(Bool, model.N_attitude_dimensions)
            mem_self = MemorisedExpression(
                att,
                mask,
                author_ID,
                true,
                ref_index,
            )
            push!(ag.memorised_self, mem_self)
        end
    end
end

function initialise_model(;
    N_agents::Int64,
    N_attitude_dimensions::Int64,
    N_identity_dimensions::Int64,
    N_memorised_other::Int64,
    N_memorised_self::Int64,
    N_search_points::Int64,
    inclusivity_scaler::Float64,
    attitude_shift_proportion::Float64,
    self_weight::Float64,
    push::Union{Float64, Nothing} = nothing,
    pull::Union{Float64, Nothing} = nothing,
    random_seed::Int64,
    sigmoid_slope_mod::Float64,
    sigmoid_slope_id::Float64,
    soft_overton_cost::Float64,
    hard_overton_cost::Float64,
    hard_overton_scalers::Vector{Float64},
    dim_probs::Vector{Float64},
    cost_increment::Float64,
    intervention::Symbol = :no_intervention, 
    intervention_time::Int64,
    cost_hike_intervention::Bool = false,
    soft_hike_intervention::Bool = false,
    init_agent_attitude_overton_proportion::Float64,
    init_agent_attitude_std::Float64,
    window_change_intervention::Bool = false,
    window_change::Float64 = 0.0,
    elite_anchor_intervention::Bool = false,
    elite_anchor_strength::Float64 = 0.5,
    equal_dimprob_intervention::Bool = false,
    track_polarization::Bool = false,
    only_esc::Bool = false
)
    my = 0.0
    hard_windows = Vector{Tuple{Float64,Float64}}()
    if length(hard_overton_scalers) != N_attitude_dimensions
        println(hard_overton_scalers, N_attitude_dimensions, length(hard_overton_scalers))
        throw("Each attitude dimension needs a hard overton scaler")
    end
    for os in hard_overton_scalers
        uplim = my + init_agent_attitude_overton_proportion * os
        lolim = my - init_agent_attitude_overton_proportion * os
        hard_overton_window = (lolim, uplim)
        push!(hard_windows, hard_overton_window)
    end
    init_agent_attitude_dist = Uniform(hard_windows[1][1] * init_agent_attitude_overton_proportion, hard_windows[1][2] * init_agent_attitude_overton_proportion)
    space = nothing
    if isnothing(push) && isnothing(pull)
        push = 0.5
        pull = 0.5
    elseif isnothing(push)
        push = 1.0 - pull
    elseif isnothing(pull)
        pull = 1.0 - push
    else
        pp = push + pull
        push = push / pp
        pull = pull / pp
    end
    mp = ModelProperties(
        N_agents,
        N_attitude_dimensions,
        N_identity_dimensions,
        N_memorised_other,
        N_memorised_self,
        N_search_points,
        inclusivity_scaler,
        attitude_shift_proportion,
        self_weight,
        pull,
        random_seed,
        sigmoid_slope_mod,
        sigmoid_slope_id,
        soft_overton_cost,
        hard_overton_cost,
        hard_windows,
        SVector{N_attitude_dimensions,Float64}(cumsum(copy(dim_probs))),
        intervention,
        intervention_time,
        cost_increment,
        cost_hike_intervention,
        soft_hike_intervention,
        window_change_intervention,
        window_change,
        elite_anchor_intervention,
        elite_anchor_strength,
        equal_dimprob_intervention,
        init_agent_attitude_dist::Distribution,
        init_agent_attitude_std,
        track_polarization,
        only_esc,
        PendingExpression[],
        Vector{SVector{N_attitude_dimensions,Float64}}(undef, N_identity_dimensions), #reference_points
        [(@MVector ones(Float64, N_agents)) for _ in 1:N_identity_dimensions], # in_ag_per_id
        [(@MVector ones(Float64, N_agents)) for _ in 1:N_identity_dimensions], # out_ag_per_id
        Tuple([(@MVector ones(Float64, N_attitude_dimensions)), (@MVector ones(Float64, N_attitude_dimensions))]), # overton_ptuple
        (@MVector zeros(Float64, N_agents)), # freq_vec
        (@MVector zeros(Float64, N_agents)), # distance_vec
        0, # no_in_context
        0, # no_out_context
        typemax(Int64), # polarized_time
        false # polarized_bool
    )
    model = StandardABM(Individual, space;
        properties = mp,
        model_step! = model_step!,
        rng = Xoshiro(random_seed)
    )
    for ag_ID in 1:N_agents
        initialise_agent!(model, ag_ID)
    end
    initialise_memories!(model)
    return model
end

function struct2model(model_params)
    ps  = model_params
    m = initialise_model(
        N_agents=ps.N_agents,
        N_attitude_dimensions=ps.N_attitude_dimensions,
        N_identity_dimensions=ps.N_identity_dimensions,
        N_memorised_other=ps.N_memorised_other,
        N_memorised_self=ps.N_memorised_self,
        init_agent_attitude_overton_proportion=ps.init_agent_attitude_overton_proportion,
        init_agent_attitude_std=ps.init_agent_attitude_std,
        N_search_points=ps.N_search_points,
        inclusivity_scaler=ps.inclusivity_scaler,
        attitude_shift_proportion=ps.attitude_shift_proportion,
        self_weight=ps.self_weight,
        random_seed=ps.random_seed,
        pull=ps.pull,
        sigmoid_slope_mod=ps.sigmoid_slope_mod,
        sigmoid_slope_id=ps.sigmoid_slope_id,
        soft_overton_cost=ps.soft_overton_cost,
        hard_overton_cost=ps.hard_overton_cost,
        hard_overton_scalers=ps.hard_overton_scalers,
        dim_probs=ps.dim_probs,
        cost_increment=ps.cost_increment,
        intervention=ps.intervention,
        intervention_time=ps.intervention_time,
        cost_hike_intervention=ps.cost_hike_intervention,
        soft_hike_intervention=ps.soft_hike_intervention,
        window_change_intervention=ps.window_change_intervention,
        window_change=ps.window_change,
        elite_anchor_intervention=ps.elite_anchor_intervention,
        elite_anchor_strength=ps.elite_anchor_strength,
        equal_dimprob_intervention=ps.equal_dimprob_intervention,
        track_polarization=ps.track_polarization,
        only_esc=ps.only_esc
    )
    return m
end

function dict2model(model_params)
    ps  = model_params
    m = initialise_model(
        N_agents=ps[:N_agents],
        N_attitude_dimensions=ps[:N_attitude_dimensions],
        N_identity_dimensions=ps[:N_identity_dimensions],
        N_memorised_other=ps[:N_memorised_other],
        N_memorised_self=ps[:N_memorised_self],
        init_agent_attitude_overton_proportion=ps[:init_agent_attitude_overton_proportion],
        init_agent_attitude_std=ps[:init_agent_attitude_std],
        N_search_points=ps[:N_search_points],
        inclusivity_scaler=ps[:inclusivity_scaler],
        attitude_shift_proportion=ps[:attitude_shift_proportion],
        self_weight=ps[:self_weight],
        random_seed=ps[:random_seed],
        pull=ps[:pull],
        sigmoid_slope_mod=ps[:sigmoid_slope_mod],
        sigmoid_slope_id=ps[:sigmoid_slope_id],
        soft_overton_cost=ps[:soft_overton_cost],
        hard_overton_cost=ps[:hard_overton_cost],
        hard_overton_scalers=ps[:hard_overton_scalers],
        dim_probs=ps[:dim_probs],
        cost_increment=ps[:cost_increment],
        intervention=ps[:intervention],
        intervention_time=ps[:intervention_time],
        cost_hike_intervention=ps[:cost_hike_intervention],
        soft_hike_intervention=ps[:soft_hike_intervention],
        window_change_intervention=ps[:window_change_intervention],
        window_change=ps[:window_change],
        equal_dimprob_intervention=ps[:equal_dimprob_intervention],
        elite_anchor_intervention=ps[:elite_anchor_intervention],
        elite_anchor_strength=ps[:elite_anchor_strength],
        track_polarization=ps[:track_polarization],
        only_esc=ps[:only_esc]
    )
    return m
end
