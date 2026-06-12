

function sample_exposed_agents(model::StandardABM, author_ID::Int64; depth_level::Int64)
    nr_exposed_dist = Binomial(N_agents - 1, 1 / (N_agents + depth_level))
    nr_exposed = rand(abmrng(model), nr_exposed_dist)
    exposed_agents_IDs = [author_ID]
    while author_ID in exposed_agents_IDs
        exposed_agents_IDs = sample(abmrng(model), 1:N_agents, nr_exposed; replace=false)
    end
    return exposed_agents_IDs
end

function sample_pending!(model::StandardABM, authors::Vector{Int64}, attitude_mask::SVector{N_attitude_dimensions,Bool}; depth_level::Int64)
    for auth in authors
        exposed_agents_IDs = sample_exposed_agents(model, auth; depth_level=depth_level) # Can build a while loop here to create all pending interactions in thread
        sampled_interaction = PendingExpression(
            auth,
            attitude_mask,
            exposed_agents_IDs,
            depth_level
        )
        sample_pending!(model, exposed_agents_IDs, attitude_mask; depth_level=depth_level + 1)
        push!(model.pending_expressions, sampled_interaction)
    end
end

function current_overton!(model::StandardABM)
    upper = model.overton_ptuple[1]
    lower = model.overton_ptuple[2]
    upper .= model[1].attitude[:,1]
    lower .= model[1].attitude[:,1]
    for ag in allagents(model)
        for id_att in eachcol(ag.attitude)
            for i in eachindex(id_att, upper, lower)
                if id_att[i] > upper[i]
                    @inbounds upper[i] = id_att[i]
                elseif id_att[i] < lower[i]
                    @inbounds lower[i] = id_att[i]
                end
            end
        end
    end
end

function util(
    model::StandardABM, 
    ref_p::MVector{N_attitude_dimensions,Float64}, 
    mask::SVector{N_attitude_dimensions,Bool}, 
    in_mean::Float64, 
    out_mean::Float64, 
    self_mean::Float64
)
    hard_overstep_cost = 0.0
    soft_overstep_cost = 0.0
    for (a, m, hw, sw_min, sw_max) in zip(ref_p, mask, model.hard_overton_windows::Vector{Tuple{Float64,Float64}}, model.overton_ptuple[2], model.overton_ptuple[1])
        if m
            if (hw[1] > a) | (a > hw[2])
                hard_overstep_cost += model.hard_overton_cost
            end
            if (sw_min > a) | (a > sw_max)
                soft_overstep_cost += model.soft_overton_cost
            end
        end
    end
    elite_anchor_cost = 0.0
    anchor_scaler = 0.0
    if model.elite_anchor_intervention
        if model.intervention == :normal
            if abmtime(model) >= model.intervention_time
                elite_anchor_cost = sqrt(sum(ref_p.^2))
                anchor_scaler = model.elite_anchor_strength
            end
        elseif model.intervention == :after
            if (typemax(Int64) > model.polarized_time) & (abmtime(model) >= (model.polarized_time + model.intervention_time))
                elite_anchor_cost = sqrt(sum(ref_p.^2))
                anchor_scaler = model.elite_anchor_strength
            end
        end
    end
    util_val = (
        (
            ((1-model.pull) * out_mean) - (model.pull * in_mean) # Social term
        ) * (1 - model.self_weight) -
        (model.self_weight * self_mean) # Self-consistency term
    ) -
    elite_anchor_cost * anchor_scaler - # Elite anchor intervention term
    hard_overstep_cost - # HW term
    soft_overstep_cost # SW term
    return util_val, hard_overstep_cost, soft_overstep_cost
end

function prob_sigmoid(d::Float64, slope::Float64)
    return 1 / (1 + exp(slope * d))
end

function get_util_n_means(
    model::StandardABM, 
    auth::Int64, 
    ref_p::MVector{N_attitude_dimensions,Float64}, 
    attitude_mask::SVector{N_attitude_dimensions,Bool}, 
    in_ag_ids::MVector{N_agents,Bool}, 
    out_ag_ids::MVector{N_agents,Bool}
)
    in_distance = 0.0
    out_distance = 0.0
    in_freq = 0.0
    out_freq = 0.0
    for exp_mem in model[auth].memorised_exposed
        if exp_mem.is_relevant
            mem_auth = exp_mem.author_ID
            @inbounds is_out = out_ag_ids[mem_auth]
            if is_out
                out_freq += 1.0
                d = euclidean(ref_p .* exp_mem.ref_index, exp_mem.attitude .* exp_mem.ref_index)
                out_distance += d
            else
                in_freq += 1.0
                d = euclidean(ref_p .* exp_mem.ref_index, exp_mem.attitude .* exp_mem.ref_index)
                in_distance += d
            end
        end
    end
    if iszero(in_freq)
        in_freq = 1.0
        model.no_in_context += 1
    end
    if iszero(out_freq)
        out_freq = 1.0
        model.no_out_context += 1
    end
    in_mean = in_distance / in_freq
    out_mean = out_distance / out_freq
    self_distance = 0.0
    for self_mem in model[auth].memorised_self
        if self_mem.is_relevant
            d = euclidean(ref_p .* self_mem.ref_index, self_mem.attitude .* self_mem.ref_index)
            self_distance += d
        end
    end
    self_mean = self_distance / model.N_memorised_self
    return util(model, ref_p, attitude_mask, in_mean, out_mean, self_mean)
end

function allocate_groups!(
    model::StandardABM, 
    ref_p::MVector{N_attitude_dimensions,Float64}, 
    auth::Int64, 
    N_memorised_exp::Int64, 
    op_dist::Int64
)
    distance_sum = 0.0
    freq_vec = abmproperties(model).freq_vec
    distance_vec = abmproperties(model).distance_vec
    freq_vec .= 0.0
    distance_vec .= 0.0
    for exp_mem in model[auth].memorised_exposed
        @inbounds d = euclidean(ref_p .* exp_mem.attitude_mask, exp_mem.attitude .* exp_mem.attitude_mask)
        @inbounds distance_vec[exp_mem.author_ID] += d
        w = d == 0.0 ? 0.0 : 1.0
        @inbounds freq_vec[exp_mem.author_ID] = w::Float64 + freq_vec[exp_mem.author_ID]::Float64
        distance_sum += d
    end
    distance_mean = distance_sum / N_memorised_exp
 
    out_ag = model.out_ag_per_id[op_dist]
    in_ag = model.in_ag_per_id[op_dist]
    out_ag .= false
    in_ag .= false
    @inbounds for i in eachindex(freq_vec, distance_vec, out_ag, in_ag)
        if !iszero(freq_vec[i])
            ag_mean = distance_vec[i] / freq_vec[i]
            greater = ag_mean > (distance_mean * model.inclusivity_scaler)
            @inbounds out_ag[i] = greater
            @inbounds in_ag[i] = !greater
        end
    end
end

function rel_intersect!(pending_mask, exp_mem)
    exp_mem.is_relevant = false
    for (i, (pm, am)) in enumerate(zip(pending_mask, exp_mem.attitude_mask))
        if pm & am
            @inbounds exp_mem.ref_index[i] = true
            exp_mem.is_relevant = true
        else
            @inbounds exp_mem.ref_index[i] = false
        end
    end
end

function get_expression(model::StandardABM, pending_exp::PendingExpression)
    for exp_mem in model[pending_exp.author_ID].memorised_exposed
        rel_intersect!(pending_exp.attitude_mask, exp_mem)
    end
    for self_mem in model[pending_exp.author_ID].memorised_self
        rel_intersect!(pending_exp.attitude_mask, self_mem)
    end
    current_overton!(model)

    # Identity choice
    auth = pending_exp.author_ID
    N_memorised_exp = length(model[auth].memorised_exposed)
    dist_utils = @MVector zeros(Float64, N_identity_dimensions)
    dist_hoc = @MVector zeros(Float64, N_identity_dimensions)
    dist_soc = @MVector zeros(Float64, N_identity_dimensions)
    for dist_i in 1:N_identity_dimensions
        @inbounds att_mu = model[auth].attitude[:,dist_i]
        @inbounds att_std = model[auth].attitude_std[:,dist_i]
        ref_p = copy(att_mu)
        abmproperties(model).reference_points[dist_i] = ref_p
        allocate_groups!(model, ref_p, auth, N_memorised_exp, dist_i)
        
        util, hoc, soc = get_util_n_means(model, auth, ref_p, pending_exp.attitude_mask, model.in_ag_per_id[dist_i], model.out_ag_per_id[dist_i])
        @inbounds dist_utils[dist_i] += util
        @inbounds dist_hoc[dist_i] += hoc
        @inbounds dist_soc[dist_i] += soc
    end
    
    model[auth].ingroup .= model.in_ag_per_id
    mean_util = mean(dist_utils)
    utils_mean_distance = dist_utils .- mean_util
    utils_mean_distance .*= -1.0
    weight_utils = @MVector zeros(Float64, N_identity_dimensions)
    weight_utils .= prob_sigmoid.(utils_mean_distance, model.sigmoid_slope_id)
    salient_dist = sample(abmrng(model), 1:N_identity_dimensions, Weights(weight_utils))

    # Self-moderation
    @inbounds stds = model[auth].attitude_std[:, salient_dist]
    @inbounds explore_dists = Normal.(model.reference_points[salient_dist], stds)
    search_points = MMatrix{N_attitude_dimensions, N_search_points, Float64}(rand(abmrng(model), exp_d) for exp_d in explore_dists, _ in 1:N_search_points)
    search_utils = @MVector zeros(Float64, N_search_points)
    search_hoc = @MVector zeros(Float64, N_search_points)
    search_soc = @MVector zeros(Float64, N_search_points)
    @inbounds for i in 1:N_search_points
        util, hoc, soc = get_util_n_means(model, auth, search_points[:,i], pending_exp.attitude_mask, model.in_ag_per_id[salient_dist], model.out_ag_per_id[salient_dist])
        search_utils[i] = util
        search_hoc[i] = hoc
        search_soc[i] = soc
    end
    @inbounds ref_util = dist_utils[salient_dist]
    @inbounds ref_hoc = dist_hoc[salient_dist]
    util_diffs = search_utils .- ref_util
    util_diffs *= -1.0
    best_search_i = argmin(util_diffs)
    @inbounds choice_hoc = search_hoc[best_search_i]
    hoc_diff = choice_hoc - ref_hoc
    hoc_diff *= -1.0
    @inbounds mod_prob = prob_sigmoid(util_diffs[best_search_i], model.sigmoid_slope_mod)
    @inbounds( 
        if rand(abmrng(model)) < mod_prob
            exp_choice = search_points[:,best_search_i]
        else
            exp_choice = model.reference_points[salient_dist]
        end
    )
    return exp_choice, salient_dist
end

function agent_expression(
    model::StandardABM,
    pending_exp::PendingExpression
)
    child_attitude, salient_dist = get_expression(model, pending_exp)
    ref_index = @MVector zeros(Bool, N_attitude_dimensions)
    expression = MemorisedExpression(
        child_attitude,
        pending_exp.attitude_mask,
        pending_exp.author_ID,
        false,
        ref_index,
    )
    return expression, salient_dist
end

function update_memory!(model::StandardABM, pending_exp::PendingExpression, memorised_exp::MemorisedExpression)
    model[pending_exp.author_ID].N_initiated += 1
    mem_self_i = mod1(model[pending_exp.author_ID].N_initiated, model.N_memorised_self)
    @inbounds model[pending_exp.author_ID].memorised_self[mem_self_i] = memorised_exp
    for exp_id in pending_exp.exposed_agents_IDs
        model[exp_id].N_interacted += 1
        mem_exp_i = mod1(model[exp_id].N_interacted, model.N_memorised_other)
        @inbounds model[exp_id].memorised_exposed[mem_exp_i] = memorised_exp
    end
end

function update_attitude!(model::StandardABM, pending_exp::PendingExpression, expression::MemorisedExpression, salient_dist::Int64)
    @inbounds step = ((expression.attitude .* pending_exp.attitude_mask) .- (model[pending_exp.author_ID].attitude[:,salient_dist] .* pending_exp.attitude_mask)) .* model.attitude_shift_proportion
    @inbounds for i in 1:N_attitude_dimensions
        model[pending_exp.author_ID].attitude[i, salient_dist] += step[i]
        if pending_exp.attitude_mask[i]
            model[pending_exp.author_ID].identity_use[i, salient_dist] = Int64(1) + model[pending_exp.author_ID].identity_use[i, salient_dist]
        end
    end
end

function pol_func(model::StandardABM)
    dim_pols = Bool[]
    for dim_i in 1:N_attitude_dimensions
        attitude = Float64[]
        for ag in allagents(model)
            atts = getindex(ag.attitude, dim_i, :)
            uses = getindex(ag.identity_use, dim_i, :)
            _, i = findmax(uses)
            push!(attitude, atts[i])
        end
        m_att = mean(attitude)
        md = mean(abs.(attitude .- m_att))
        low_att = attitude[attitude .< m_att]
        high_att = attitude[attitude .>= m_att]
        md_low = mean(abs.(low_att .- mean(low_att)))
        md_high = mean(abs.(high_att .- mean(high_att)))
        dim_pol = ((md_low / md) < 0.3) && ((md_high / md) < 0.3)
        push!(dim_pols, dim_pol)
    end
    return any(dim_pols)
end

function esc_func(model::StandardABM)
    wind_size = model.hard_overton_windows[1][2] - model.hard_overton_windows[1][1]
    half_wind_size = wind_size / 2
    init_mean = 0.0
    hw_low = init_mean .- half_wind_size .- 1
    hw_high = init_mean .+ half_wind_size .+ 1
    escape = Bool[]
    escape_threshold = 0.05
    for ag in allagents(model)
        escape_bool = (ag.attitude .< hw_low) .| (ag.attitude .> hw_high)
        both_dim_esc = any(escape_bool, dims=2)
        escape = [escape; both_dim_esc]
    end
    return mean(escape) > escape_threshold
end

function state_check(model::StandardABM, curr_time::Int64)
    if (curr_time % 100 == 0)
        pol_bool = pol_func(model)
        esc_bool = esc_func(model)
        if model.only_esc
            return esc_bool
        else
            return pol_bool | esc_bool
        end
    else
        return false
    end
end

function intervention_func!(model::StandardABM)
    if model.cost_hike_intervention
            abmproperties(model).hard_overton_cost = model.hard_overton_cost + model.cost_increment
    end
    if model.soft_hike_intervention
            abmproperties(model).soft_overton_cost = model.soft_overton_cost + model.cost_increment
    end
    if model.window_change_intervention
            for i in eachindex(model.hard_overton_windows)
                model.hard_overton_windows[i] = (-model.window_change/2, model.window_change/2)
            end
    end
    if model.equal_dimprob_intervention
            model.dim_probs = cumsum(1.0 / N_attitude_dimensions for _ in 1:N_attitude_dimensions)
    end
end

function tracking_func!(model::StandardABM)
    curr_t = abmtime(model)
    if model.polarized_time > curr_t # Check if population is in a polarized state
        if curr_t % 100 == 0
            t_func_bool = state_check(model, curr_t)
            if t_func_bool
                model.polarized_time = curr_t
            end
        end
    elseif (model.polarized_time + 500_000) < curr_t # Check that the polarized state is maintained
        if curr_t % 100 == 0
            t_func_bool = state_check(model, curr_t)
            model.polarized_bool = t_func_bool
        end
    end
end

function model_step!(model::StandardABM)
    if length(model.pending_expressions) == 0
        author_IDs = rand(abmrng(model), 1:N_agents, 1)
        ref_rand = rand(abmrng(model))
        dim = round(sum(ref_rand .> model.dim_probs) + 1, digits=0)
        attitude_mask = (@SVector [dim == d ? true : false for d in 1:N_attitude_dimensions])
        sample_pending!(model, author_IDs, attitude_mask; depth_level=0)  # A sampled interaction is always a root interaction
    end
    pending_exp = last(model.pending_expressions)
    expression, salient_dist = agent_expression(model, pending_exp)
    update_memory!(model, pending_exp, expression)
    update_attitude!(model, pending_exp, expression, salient_dist)
    resize!(model.pending_expressions, length(model.pending_expressions) - 1)
    
    if model.intervention == :normal
        if abmtime(model) == model.intervention_time
            intervention_func!(model)
        end
    elseif model.intervention == :after
        tracking_func!(model)
        if abmtime(model) == (model.polarized_time + model.intervention_time)
            intervention_func!(model)
        end
    end
    if model.track_polarization
        tracking_func!(model)
    end
end
