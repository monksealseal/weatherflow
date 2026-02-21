# ---------------------------------------------------------------------------
# Mass budget and conservation diagnostics
# ---------------------------------------------------------------------------

"""
    compute_mass_budget(conc, species_list, hgrid, vgrid, ps)

Compute the global mass of each species [kg].
Returns a Dict mapping species name to total mass.
"""
function compute_mass_budget(conc::AbstractArray{Float64, 4},
                              species_list::Vector{Species},
                              hgrid::HorizontalGrid,
                              vgrid::VerticalGrid,
                              ps::AbstractMatrix)
    nlon, nlat, nlevels, nspec = size(conc)
    dp = layer_thickness_dp(vgrid, ps)
    p = pressure_at_levels(vgrid, ps)

    budget = Dict{Symbol, Float64}()

    conc_cpu = Array(conc)
    dp_cpu = Array(dp)
    p_cpu = Array(p)
    area = Array(hgrid.area)

    for (s, sp) in enumerate(species_list)
        total_mass = 0.0

        for k in 1:nlevels, j in 1:nlat, i in 1:nlon
            # Number density [molec cm⁻³] → mass concentration [kg m⁻³]
            # ρ_s = n × mw / Na × 1e6 (cm⁻³ → m⁻³)
            n = max(conc_cpu[i, j, k, s], 0.0)
            rho_s = n * sp.mw / Constants.N_A * 1e6  # kg m⁻³

            # Layer thickness [m] from pressure thickness
            T_approx = 250.0  # Rough average temperature
            p_k = p_cpu[i, j, k]
            rho_air = p_k / (Constants.R_GAS / Constants.M_AIR * T_approx)
            dz = dp_cpu[i, j, k] / (rho_air * Constants.G)

            # Mass in this cell [kg]
            total_mass += rho_s * dz * area[i, j]
        end

        budget[sp.name] = total_mass
    end

    return budget
end

"""
    print_budget_summary(budget; prev_budget=nothing)

Print a formatted summary of the mass budget.
If `prev_budget` is provided, also shows change rates.
"""
function print_budget_summary(budget::Dict{Symbol, Float64};
                               prev_budget::Union{Dict{Symbol, Float64}, Nothing}=nothing,
                               dt_hours::Float64=0.0)
    println("┌────────────┬───────────────────┬──────────────┐")
    println("│  Species   │    Mass (Tg)      │  Change/hr   │")
    println("├────────────┼───────────────────┼──────────────┤")

    for (sp, mass) in sort(collect(budget), by=x->x[1])
        mass_tg = mass / 1e9  # kg → Tg

        if prev_budget !== nothing && haskey(prev_budget, sp) && dt_hours > 0
            prev_mass_tg = prev_budget[sp] / 1e9
            change_rate = (mass_tg - prev_mass_tg) / dt_hours
            @printf("│ %-10s │ %15.4f   │ %+10.4f   │\n",
                    sp, mass_tg, change_rate)
        else
            @printf("│ %-10s │ %15.4f   │     N/A      │\n", sp, mass_tg)
        end
    end

    println("└────────────┴───────────────────┴──────────────┘")
end

"""
    check_conservation(budget_old, budget_new, species; tolerance=0.01)

Check mass conservation for a species. Returns true if the relative
change is within the specified tolerance.
"""
function check_conservation(budget_old::Dict{Symbol, Float64},
                            budget_new::Dict{Symbol, Float64},
                            species::Symbol;
                            tolerance::Float64=0.01)
    if !haskey(budget_old, species) || !haskey(budget_new, species)
        return true  # Can't check
    end

    m_old = budget_old[species]
    m_new = budget_new[species]

    if m_old == 0.0
        return m_new == 0.0
    end

    rel_change = abs(m_new - m_old) / m_old
    if rel_change > tolerance
        @warn "Mass conservation violation for $species: " *
              "$(rel_change * 100)% change (tolerance: $(tolerance * 100)%)"
        return false
    end
    return true
end

"""
    global_mean_concentration(conc, species_idx, hgrid, vgrid, ps; level=nothing)

Compute the area-weighted global mean concentration for a species.
If `level` is specified, return the mean at that level only.
"""
function global_mean_concentration(conc::AbstractArray{Float64, 4},
                                    species_idx::Int,
                                    hgrid::HorizontalGrid;
                                    level::Union{Int, Nothing}=nothing)
    nlon, nlat, nlevels, _ = size(conc)
    area = Array(hgrid.area)

    total_conc = 0.0
    total_area = 0.0

    conc_cpu = Array(conc)

    if level !== nothing
        for j in 1:nlat, i in 1:nlon
            total_conc += conc_cpu[i, j, level, species_idx] * area[i, j]
            total_area += area[i, j]
        end
    else
        for k in 1:nlevels, j in 1:nlat, i in 1:nlon
            total_conc += conc_cpu[i, j, k, species_idx] * area[i, j]
            total_area += area[i, j]
        end
        total_area *= nlevels  # Normalise over all levels
    end

    return total_area > 0 ? total_conc / total_area : 0.0
end
