function run_proportion_escaped(escape; escape_threshold=0.05)
    return mean(escape) > escape_threshold
end

function run_bimodal_max_use(attitude, ismax, ratio_threshold=0.3)
    rel_atts = attitude[ismax]
    m_att = mean(rel_atts)
    md = mean(abs.(rel_atts .- m_att))
    low_att = rel_atts[rel_atts .< m_att]
    high_att = rel_atts[rel_atts .>= m_att]
    md_low = mean(abs.(low_att .- mean(low_att)))
    md_high = mean(abs.(high_att .- mean(high_att)))
    low_bool = (md_low / md) < ratio_threshold
    high_bool = (md_high / md) < ratio_threshold
    return low_bool && high_bool
end

function latent_extremism(attitude, identity_use, window_width)
    identity_props = identity_use ./ sum(identity_use)
    outside_attitude = abs.(attitude) .- (window_width ./ 2)
    return mean(outside_attitude .* (1.0 .- identity_props))
end

function run_state_classification_cluster(raw_df; init_mean=0.0, polarised_threshold=0.3, escape_threshold=0.05, drop_threshold=0.02)
    local_df = copy(raw_df)

    # Flatten the data frame to have one row per attitude dimension and opinion distribution
    local_df.identity_nr .= [[i for i in 1:N_od for _ in 1:N_ad] for (N_od, N_ad) in zip(local_df.N_identity_dimensions, local_df.N_attitude_dimensions)]
    local_df.dimension_nr .= [[i for _ in 1:N_od for i in 1:N_ad] for (N_od, N_ad) in zip(local_df.N_identity_dimensions, local_df.N_attitude_dimensions)]
    perm_vars = [:attitude, :identity_nr, :dimension_nr, :identity_use, :wind_size, :random_seed, :id, :time]
    flat_df = flatten(local_df[:,perm_vars], [:attitude, :identity_use, :identity_nr, :dimension_nr])
    local_df = nothing

    # Indicate for each attitude if it has escaped the hard Overton window (with margin)
    hw_low = init_mean .- (flat_df.wind_size ./ 2)
    hw_high = init_mean .+ (flat_df.wind_size ./ 2)
    flat_df.margin_escape = (flat_df.attitude .< (hw_low .- 1)) .| (flat_df.attitude .> (hw_high .+ 1))
    
    # Calculate the number of times each attitude of each identity has been used since the second to last saved model state (200k steps in our simulations)
    sort!(flat_df, :time)
    per_identity_vars = [:random_seed, :dimension_nr, :id, :identity_nr]
    gdf = groupby(flat_df, per_identity_vars)
    identity_df = combine(gdf, 
        :identity_use => (x -> x[end] .- x[end-1]) => :identity_use_intervention_time  
    )
    flat_df = flat_df[flat_df.time .== maximum(flat_df.time), :]
    flat_df = leftjoin(flat_df, identity_df, on=per_identity_vars)

    # Identifying the identity used the most per individual and attitude dimension and which attitudes have escaped the hard Overton window 
    # despite being used drop_threshold proportion of the time since the last measured model state
    per_individual_vars = [:random_seed, :dimension_nr, :id]
    gdf = groupby(flat_df, per_individual_vars)
    individual_df = combine(gdf, 
        :identity_use_intervention_time => sum => :identity_use_intervention_time_sum,
        :identity_use => maximum => :identity_use_max,
    )
    flat_df = leftjoin(flat_df, individual_df, on=per_individual_vars)
    individual_df = nothing
    flat_df.ismax = (flat_df.identity_use .== flat_df.identity_use_max)
    flat_df.identity_use_intervention_time_prop = flat_df.identity_use_intervention_time ./ flat_df.identity_use_intervention_time_sum
    flat_df.escape_intervention = flat_df.margin_escape .& (flat_df.identity_use_intervention_time_prop .> drop_threshold)
    
    # Evaluating if the conditions for runaway polarization (escaped) and contained polarization (bimodal) are met
    # per attitude dimension
    gdf = groupby(flat_df, [:random_seed, :dimension_nr])
    attitude_dimension_df = combine(gdf, 
        :escape_intervention => (endi -> run_proportion_escaped(endi, escape_threshold=escape_threshold)) => :escaped_intervention_prop_bool,
        [:attitude, :ismax] => ((att, imax) -> run_bimodal_max_use(att, imax)) => :bimodal_max_bool,
    )

    # Calculating latent extremism per individual and attitude dimension
    gdf = groupby(flat_df, per_individual_vars)
    individual_df = combine(gdf, 
        [:attitude, :identity_use_intervention_time, :wind_size] => ((att, idu_it, ws) -> latent_extremism(att, idu_it, ws)) => :latent_extremism,
    )
    flat_df = nothing
    gdf = groupby(individual_df, [:random_seed, :dimension_nr])
    latent_extremism_df = combine(gdf, 
        :latent_extremism => mean => :latent_extremism,
    )
    attitude_dimension_df = leftjoin(attitude_dimension_df, latent_extremism_df, on=[:random_seed, :dimension_nr])
    individual_df = nothing

    # Final classification of states (consensus, contained, runaway) per attitude dimension
    attitude_dimension_df.collapsed_prop_bool = .~attitude_dimension_df.escaped_intervention_prop_bool .& .~attitude_dimension_df.bimodal_max_bool
    attitude_dimension_df.bimodal_max_prop_bool = attitude_dimension_df.bimodal_max_bool .& .~attitude_dimension_df.escaped_intervention_prop_bool
    return attitude_dimension_df
end

function run_state_aggregation_cluster(df; assume_2_dims=true)

    # State classification aggregation over attitude dimensions
    gdf = groupby(df, :random_seed)
    cpdf = combine(gdf, 
        :collapsed_prop_bool => all => :collapsed_prop,
        :escaped_intervention_prop_bool => any => :escaped_prop,
        :bimodal_max_prop_bool => any => :bimodal_prop,
    )
    cpdf.bimodal_prop .= cpdf.bimodal_prop .& .~cpdf.escaped_prop
    outcome_dict = Dict()
    outcome_dict[:consensus_prop] = mean(cpdf.collapsed_prop)
    outcome_dict[:runaway_prop] = mean(cpdf.escaped_prop)
    outcome_dict[:contained_prop] = mean(cpdf.bimodal_prop)
    outcome_dict[:consensus_prop_std] = std(cpdf.collapsed_prop)
    outcome_dict[:runaway_prop_std] = std(cpdf.escaped_prop)
    outcome_dict[:contained_prop_std] = std(cpdf.bimodal_prop)
    outcome_dict[:outcome_state] = "No majority"
    outcome_label_dict = Dict(
        :consensus_prop => "Consensus", 
        :runaway_prop => "Runaway polarization", 
        :contained_prop => "Contained polarization",
    )
    for k in [:consensus_prop, :runaway_prop, :contained_prop]
        if outcome_dict[k] > 0.5
            outcome_dict[:outcome_state] = outcome_label_dict[k]
        end
    end
    if assume_2_dims
        dim1_df = df[df.dimension_nr .== 1, :]
        dim2_df = df[df.dimension_nr .== 2, :]  
        outcome_dict[:latent_extremism_mean_dim1] = mean(dim1_df.latent_extremism)
        outcome_dict[:latent_extremism_std_dim1] = std(dim1_df.latent_extremism)
        outcome_dict[:latent_extremism_mean_dim2] = mean(dim2_df.latent_extremism)
        outcome_dict[:latent_extremism_std_dim2] = std(dim2_df.latent_extremism)
    end
    return outcome_dict
end

function data_prep(agent_data_path, model_data_path; assume_2_dims=true)
    adf = Arrow.Table(agent_data_path) |> DataFrame
    mdf = Arrow.Table(model_data_path) |> DataFrame
    mdf.wind_size = last.(getindex.(mdf.hard_overton_windows, 1)) .- first.(getindex.(mdf.hard_overton_windows, 1))
    df = leftjoin(adf, mdf[:,Not(:time)], on=[:ensemble])
    raw_df = run_state_classification_cluster(df)
    agg_dict = run_state_aggregation_cluster(raw_df, assume_2_dims=assume_2_dims)
    return agg_dict
end