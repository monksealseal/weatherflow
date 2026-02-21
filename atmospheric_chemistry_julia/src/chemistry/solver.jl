# ---------------------------------------------------------------------------
# Chemistry solver — implicit Euler / Rosenbrock for stiff ODE system
# ---------------------------------------------------------------------------

"""
    solve_chemistry!(concentrations, mechanism, T, M, j_rates, dt;
                     method=:implicit_euler, n_substeps=4)

Solve the chemical kinetics at a single grid cell or across the full grid.

Arguments:
- `concentrations`: 4-D array (nlon, nlat, nlevels, n_species) [molec cm⁻³]
- `mechanism`: `ChemicalMechanism`
- `T`: 3-D temperature array [K]
- `M`: 3-D air number density [molec cm⁻³]
- `j_rates`: 4-D photolysis rates (nlon, nlat, nlevels, n_photo_rxns)
- `dt`: Chemistry timestep [s]

Modifies `concentrations` in-place.
"""
function solve_chemistry!(conc::AbstractArray{Float64, 4},
                           mech::ChemicalMechanism,
                           T::AbstractArray{Float64, 3},
                           M::AbstractArray{Float64, 3},
                           j_rates::AbstractArray{Float64, 4},
                           dt::Float64;
                           n_substeps::Int=4)
    nlon, nlat, nlevels, nspec = size(conc)
    dt_sub = dt / n_substeps

    # Map photolysis reaction indices
    photo_map = _build_photolysis_map(mech)

    Threads.@threads for idx in CartesianIndices((nlon, nlat, nlevels))
        i, j, k = Tuple(idx)

        T_local = T[i, j, k]
        M_local = M[i, j, k]

        # Extract local concentrations
        c = zeros(Float64, nspec)
        for s in 1:nspec
            c[s] = max(conc[i, j, k, s], 0.0)
        end

        # Local j-values
        j_local = zeros(Float64, 4)
        for p in 1:min(4, size(j_rates, 4))
            j_local[p] = j_rates[i, j, k, p]
        end

        # Sub-step integration
        for _ in 1:n_substeps
            _euler_implicit_step!(c, mech, T_local, M_local,
                                  j_local, photo_map, dt_sub)
        end

        # Write back
        for s in 1:nspec
            conc[i, j, k, s] = max(c[s], 0.0)
        end
    end
end

"""
Build a mapping from photolysis reaction index to j_rates array index.
"""
function _build_photolysis_map(mech::ChemicalMechanism)
    photo_labels = Dict(
        "NO2 + hv -> NO + O3"   => 1,
        "O3 + hv -> 2OH (net)"  => 2,
        "H2O2 + hv -> 2OH"      => 3,
        "HCHO + hv -> CO + 2HO2" => 4,
    )

    pmap = Dict{Int, Int}()
    for (r, rxn) in enumerate(mech.reactions)
        if rxn.rtype == PHOTOLYSIS
            idx = get(photo_labels, rxn.label, 0)
            if idx > 0
                pmap[r] = idx
            end
        end
    end
    return pmap
end

"""
Single implicit Euler step for the chemical ODE system.

Uses a first-order linearisation: c^{n+1} = c^n + dt × P/(1 + dt × L)
where P = production rate, L = loss frequency.
This is unconditionally stable for any dt.
"""
function _euler_implicit_step!(c::Vector{Float64},
                                mech::ChemicalMechanism,
                                T::Float64, M::Float64,
                                j_local::Vector{Float64},
                                photo_map::Dict{Int, Int},
                                dt::Float64)
    nspec = mech.n_species
    production = zeros(Float64, nspec)
    loss_freq  = zeros(Float64, nspec)   # [s⁻¹]

    # Species index lookup
    sp_idx = Dict(mech.species[s].name => s for s in 1:nspec)

    for (r, rxn) in enumerate(mech.reactions)
        # Compute rate constant
        if rxn.rtype == PHOTOLYSIS
            k = get(photo_map, r, 0) > 0 ? j_local[photo_map[r]] : 0.0
        else
            k = rate_constant(rxn, T, M)
        end

        # Compute reaction rate [molec cm⁻³ s⁻¹]
        rate = k
        for (ri, rname) in enumerate(rxn.reactants)
            si = get(sp_idx, rname, 0)
            if si > 0
                rate *= c[si]^rxn.stoich_r[ri]
            end
        end

        # Accumulate production and loss
        for (ri, rname) in enumerate(rxn.reactants)
            si = get(sp_idx, rname, 0)
            if si > 0
                # Loss frequency = rate / concentration (avoid division by zero)
                if c[si] > 1.0  # at least 1 molec/cm³
                    loss_freq[si] += rxn.stoich_r[ri] * rate / c[si]
                end
            end
        end

        for (pi, pname) in enumerate(rxn.products)
            si = get(sp_idx, pname, 0)
            if si > 0
                production[si] += rxn.stoich_p[pi] * rate
            end
        end
    end

    # Implicit Euler update: c_new = (c + dt × P) / (1 + dt × L)
    for s in 1:nspec
        c[s] = (c[s] + dt * production[s]) / (1.0 + dt * loss_freq[s])
        c[s] = max(c[s], 0.0)  # Positivity constraint
    end
end

"""
    ppb_to_number_density(ppb, T, p)

Convert mixing ratio [ppb] to number density [molec cm⁻³].
"""
function ppb_to_number_density(ppb::Float64, T::Float64, p::Float64)
    # n = p / (kB × T) × ppb × 1e-9
    n_air = p / (Constants.k_B * T) * 1e-6  # molec cm⁻³
    return n_air * ppb * 1e-9
end

"""
    number_density_to_ppb(n, T, p)

Convert number density [molec cm⁻³] to mixing ratio [ppb].
"""
function number_density_to_ppb(n::Float64, T::Float64, p::Float64)
    n_air = p / (Constants.k_B * T) * 1e-6
    return n / n_air * 1e9
end

"""
    air_number_density(T, p)

Compute air number density [molec cm⁻³] from temperature [K] and pressure [Pa].
"""
function air_number_density(T::Float64, p::Float64)
    return p / (Constants.k_B * T) * 1e-6  # m⁻³ → cm⁻³
end
