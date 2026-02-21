# ---------------------------------------------------------------------------
# Main model driver — configuration, state, and time integration
# ---------------------------------------------------------------------------

"""
    ModelConfig(; kwargs...)

Configuration for the atmospheric chemistry transport model.
"""
Base.@kwdef struct ModelConfig
    # Grid
    resolution      :: Float64 = 1.0          # Horizontal resolution [degrees]
    nlevels         :: Int     = 47           # Number of vertical levels
    p_top           :: Float64 = 1.0          # Model top pressure [Pa]

    # Time
    start_date      :: DateTime = DateTime(2023, 1, 1)
    end_date        :: DateTime = DateTime(2023, 1, 2)
    dt_advection    :: Float64 = 900.0        # Advection timestep [s] (15 min)
    dt_chemistry    :: Float64 = 600.0        # Chemistry timestep [s] (10 min)
    dt_diffusion    :: Float64 = 900.0        # Diffusion timestep [s]
    dt_emission     :: Float64 = 3600.0       # Emission update interval [s]
    met_update_freq :: Int     = 6            # Meteorology update frequency [hours]

    # Data
    era5_data_dir   :: String  = "data/era5"
    edgar_data_dir  :: String  = "data/edgar"
    edgar_version   :: String  = "v8.0"
    edgar_year      :: Int     = 2022

    # Output
    output_dir      :: String  = "output"
    output_freq_h   :: Int     = 6            # Output frequency [hours]
    output_prefix   :: String  = "actm"

    # Numerics
    chem_substeps   :: Int     = 4            # Chemistry sub-steps per dt_chemistry
    gpu             :: Bool    = false         # Enable GPU acceleration

    # Physics switches
    do_advection    :: Bool    = true
    do_chemistry    :: Bool    = true
    do_diffusion    :: Bool    = true
    do_convection   :: Bool    = true
    do_dry_dep      :: Bool    = true
    do_wet_dep      :: Bool    = true
    do_emissions    :: Bool    = true
end

"""
    ModelState

Holds the full model state during integration.
"""
mutable struct ModelState
    config     :: ModelConfig
    hgrid      :: HorizontalGrid
    vgrid      :: VerticalGrid
    mechanism  :: ChemicalMechanism
    gpub       :: GPUBackend

    # Concentration array: (nlon, nlat, nlevels, nspecies) [molec cm⁻³]
    conc       :: AbstractArray{Float64, 4}

    # Meteorological fields (updated from ERA5)
    met        :: NamedTuple

    # Emission fields (2-D, surface) — Dict from species symbol to array
    emissions  :: Dict{Symbol, AbstractMatrix{Float64}}

    # Photolysis rates: (nlon, nlat, nlevels, 4)
    j_rates    :: AbstractArray{Float64, 4}

    # Air number density: (nlon, nlat, nlevels) [molec cm⁻³]
    M_air      :: AbstractArray{Float64, 3}

    # Diagnostics
    diag       :: DiagnosticOutput
    budget_prev :: Dict{Symbol, Float64}

    # Time tracking
    current_time :: DateTime
    step_count   :: Int
end

"""
    ModelState(config::ModelConfig)

Construct and initialise the model state from a configuration.
"""
function ModelState(config::ModelConfig)
    gpu = config.gpu

    @info "Initialising atmospheric chemistry transport model..."
    @info "  Resolution: $(config.resolution)°"
    @info "  Levels: $(config.nlevels)"
    @info "  Period: $(config.start_date) → $(config.end_date)"
    @info "  GPU: $(gpu ? "enabled" : "disabled")"

    # Build grids
    hgrid = HorizontalGrid(; resolution=config.resolution, gpu=gpu)
    vgrid = VerticalGrid(; nlevels=config.nlevels, p_top=config.p_top, gpu=gpu)
    @info "  Grid: $(hgrid)"

    # Build chemistry
    mechanism = default_mechanism()
    @info "  Chemistry: $(mechanism.n_species) species, $(mechanism.n_reactions) reactions"

    # GPU backend
    gpub = GPUBackend(; gpu=gpu)

    # Allocate concentration array
    nspec = mechanism.n_species
    conc = device_array(Float64, hgrid.nlon, hgrid.nlat, vgrid.nlevels, nspec; gpu=gpu)

    # Initialise with background mixing ratios
    _initialise_concentrations!(conc, mechanism.species, vgrid, hgrid)

    # Placeholder meteorological fields
    met = _default_met(hgrid, vgrid, gpu)

    # Photolysis rates
    j_rates = device_array(Float64, hgrid.nlon, hgrid.nlat, vgrid.nlevels, 4; gpu=gpu)

    # Air number density
    M_air = device_array(Float64, hgrid.nlon, hgrid.nlat, vgrid.nlevels; gpu=gpu)
    _compute_M_air!(M_air, met.T, vgrid, met.ps)

    # Load emissions
    emissions = _load_all_emissions(config, mechanism.species, hgrid, gpu)

    # Diagnostics
    diag = DiagnosticOutput(; output_dir=config.output_dir,
                             prefix=config.output_prefix,
                             species_list=mechanism.species,
                             hgrid=hgrid, vgrid=vgrid,
                             output_freq_h=config.output_freq_h)

    return ModelState(config, hgrid, vgrid, mechanism, gpub,
                      conc, met, emissions, j_rates, M_air, diag,
                      Dict{Symbol, Float64}(),
                      config.start_date, 0)
end

"""
    run_simulation!(state::ModelState)

Main time integration loop. Uses operator splitting:
1. Update meteorology (from ERA5)
2. Emissions injection
3. Advection
4. Vertical diffusion
5. Convective transport
6. Chemistry (with photolysis)
7. Dry deposition
8. Wet deposition
9. Diagnostics output
"""
function run_simulation!(state::ModelState)
    config = state.config
    t = config.start_date
    t_end = config.end_date
    total_hours = Dates.value(t_end - t) / 3_600_000

    @info "Starting simulation: $t → $t_end ($(total_hours) hours)"

    # Initial budget
    state.budget_prev = compute_mass_budget(state.conc, state.mechanism.species,
                                             state.hgrid, state.vgrid, state.met.ps)
    @info "Initial mass budget:"
    print_budget_summary(state.budget_prev)

    # Write initial state
    write_diagnostics(state.diag, state.conc, state.met, t)

    # Determine the finest timestep
    dt_min = minimum([config.dt_advection, config.dt_chemistry, config.dt_diffusion])
    n_steps = ceil(Int, Dates.value(t_end - t) / (dt_min * 1000))

    @info "Integration: $n_steps steps of $(dt_min)s"

    met_manager = ERA5DataManager(; data_dir=config.era5_data_dir,
                                   resolution=config.resolution)

    prog = Progress(n_steps; desc="Simulating: ", showspeed=true)

    step = 0
    while t < t_end
        step += 1
        state.step_count = step
        state.current_time = t

        # === 1. Update meteorology ===
        hour = Dates.hour(t)
        if step == 1 || (hour % config.met_update_freq == 0 && Dates.minute(t) == 0 && Dates.second(t) == 0)
            _update_meteorology!(state, met_manager, t)
        end

        # === 2. Emissions ===
        if config.do_emissions
            _inject_emissions!(state, dt_min)
        end

        # === 3. Advection ===
        if config.do_advection
            advect!(state.conc, state.met.u, state.met.v, state.met.w,
                    state.hgrid, state.vgrid, state.met.ps, config.dt_advection;
                    gpu=config.gpu)
        end

        # === 4. Vertical diffusion ===
        if config.do_diffusion
            blh = state.met.blh
            Kz = compute_Kz(Array(blh), state.vgrid, Array(state.met.ps))
            if config.gpu
                Kz = to_device(Kz; gpu=true)
            end
            diffuse_vertical!(state.conc, Kz, state.vgrid,
                              state.met.ps, config.dt_diffusion)
        end

        # === 5. Convective transport ===
        if config.do_convection
            convective_transport!(state.conc, state.met.T, state.met.q,
                                  state.met.ps, state.vgrid, dt_min)
        end

        # === 6. Photolysis & Chemistry ===
        if config.do_chemistry
            # Update photolysis rates
            o3_idx = species_index(state.mechanism.species, :O3)
            o3_column = _compute_o3_column(state.conc, o3_idx,
                                            state.vgrid, state.met.ps)
            cloud_frac = _estimate_cloud_fraction(state.met.q, state.met.T,
                                                   state.vgrid, state.met.ps)

            compute_photolysis_field!(Array(state.j_rates), state.hgrid, state.vgrid,
                                      Array(state.met.ps), o3_column,
                                      cloud_frac, t)

            # Solve chemistry
            solve_chemistry!(Array(state.conc), state.mechanism,
                             Array(state.met.T), Array(state.M_air),
                             Array(state.j_rates), config.dt_chemistry;
                             n_substeps=config.chem_substeps)
        end

        # === 7. Dry deposition ===
        if config.do_dry_dep
            dry_deposition!(state.conc, state.mechanism.species,
                            state.hgrid, state.vgrid, state.met.ps,
                            state.met.u10, state.met.v10, state.met.blh,
                            dt_min)
        end

        # === 8. Wet deposition ===
        if config.do_wet_dep
            wet_deposition!(state.conc, state.mechanism.species,
                            state.met.precip, state.hgrid, state.vgrid,
                            state.met.ps, state.met.T, dt_min)
        end

        # === 9. Diagnostics ===
        t += Dates.Second(round(Int, dt_min))

        elapsed_h = Dates.value(t - config.start_date) / 3_600_000
        if elapsed_h > 0 && mod(elapsed_h, config.output_freq_h) ≈ 0
            budget_new = compute_mass_budget(state.conc, state.mechanism.species,
                                              state.hgrid, state.vgrid, state.met.ps)
            print_budget_summary(budget_new; prev_budget=state.budget_prev,
                                 dt_hours=Float64(config.output_freq_h))
            state.budget_prev = budget_new

            write_diagnostics(state.diag, state.conc, state.met, t)
        end

        next!(prog)
    end

    @info "Simulation complete: $(state.step_count) steps"

    # Final budget
    budget_final = compute_mass_budget(state.conc, state.mechanism.species,
                                        state.hgrid, state.vgrid, state.met.ps)
    @info "Final mass budget:"
    print_budget_summary(budget_final)

    return state
end

# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

"""Initialise concentrations from background mixing ratios."""
function _initialise_concentrations!(conc, species_list, vgrid, hgrid)
    conc_cpu = Array(conc)
    nlon, nlat, nlevels, nspec = size(conc_cpu)

    for (s, sp) in enumerate(species_list)
        if sp.initial_ppb > 0
            for k in 1:nlevels, j in 1:nlat, i in 1:nlon
                # Approximate pressure and temperature
                η = k / nlevels
                p = Constants.P_REF * η
                T = Constants.T_REF - 60.0 * (1 - η)  # Rough lapse rate

                # Convert ppb to number density
                conc_cpu[i, j, k, s] = ppb_to_number_density(
                    sp.initial_ppb, T, p)
            end
        end
    end

    if typeof(conc) != typeof(conc_cpu)
        conc .= to_device(conc_cpu; gpu=true)
    else
        conc .= conc_cpu
    end
end

"""Create default meteorological fields (isothermal atmosphere with gentle zonal flow)."""
function _default_met(hgrid, vgrid, gpu)
    nlon, nlat, nlevels = hgrid.nlon, hgrid.nlat, vgrid.nlevels

    T = zeros(Float64, nlon, nlat, nlevels)
    u = zeros(Float64, nlon, nlat, nlevels)
    v = zeros(Float64, nlon, nlat, nlevels)
    w = zeros(Float64, nlon, nlat, nlevels)
    q = zeros(Float64, nlon, nlat, nlevels)
    ps = fill(Constants.P_REF, nlon, nlat)
    blh = fill(1000.0, nlon, nlat)
    precip = zeros(Float64, nlon, nlat)
    ssrd = fill(200.0, nlon, nlat)
    u10 = fill(3.0, nlon, nlat)
    v10 = fill(1.0, nlon, nlat)
    t2m = fill(288.0, nlon, nlat)

    for k in 1:nlevels, j in 1:nlat, i in 1:nlon
        η = k / nlevels
        T[i, j, k] = 220.0 + 68.0 * η  # Simple temperature profile
        lat = hgrid.lat[j]

        # Zonal wind: mid-latitude jet
        u[i, j, k] = 10.0 * sin(deg2rad(lat))^2 * sin(π * (1 - η))
        # Weak meridional flow
        v[i, j, k] = 0.5 * sin(2 * deg2rad(lat)) * η
        # Very weak vertical motion
        w[i, j, k] = -0.01 * sin(π * η) * cos(deg2rad(lat))

        # Specific humidity (decreasing with altitude)
        q[i, j, k] = 0.01 * η^3 * exp(-((lat - 10)^2) / (2 * 30^2))
    end

    if gpu
        return (T=to_device(T; gpu), u=to_device(u; gpu), v=to_device(v; gpu),
                w=to_device(w; gpu), q=to_device(q; gpu),
                ps=to_device(ps; gpu), blh=to_device(blh; gpu),
                precip=to_device(precip; gpu), ssrd=to_device(ssrd; gpu),
                u10=to_device(u10; gpu), v10=to_device(v10; gpu),
                t2m=to_device(t2m; gpu))
    end

    return (T=T, u=u, v=v, w=w, q=q, ps=ps, blh=blh,
            precip=precip, ssrd=ssrd, u10=u10, v10=v10, t2m=t2m)
end

"""Compute air number density at all grid points."""
function _compute_M_air!(M_air, T, vgrid, ps)
    M_cpu = Array(M_air)
    T_cpu = Array(T)
    ps_cpu = Array(ps)
    nlon, nlat, nlevels = size(M_cpu)

    for k in 1:nlevels, j in 1:nlat, i in 1:nlon
        p_k = vgrid.a_full[k] + vgrid.b_full[k] * ps_cpu[i, j]
        M_cpu[i, j, k] = air_number_density(T_cpu[i, j, k], p_k)
    end

    M_air .= (typeof(M_air) != typeof(M_cpu)) ? to_device(M_cpu; gpu=true) : M_cpu
end

"""Load emissions for all species that have them."""
function _load_all_emissions(config, species_list, hgrid, gpu)
    mgr = EDGARManager(; data_dir=config.edgar_data_dir,
                        version=config.edgar_version,
                        base_year=config.edgar_year,
                        resolution=config.resolution)

    # Download (or generate synthetic) emissions
    emitting_species = [sp.name for sp in species_list if sp.has_emissions]
    download_edgar!(mgr; species=emitting_species)

    emissions = Dict{Symbol, AbstractMatrix{Float64}}()
    for sp in species_list
        if sp.has_emissions
            # Map model species to EDGAR species
            edgar_sp = sp.name
            if sp.name == :NO || sp.name == :NO2
                edgar_sp = :NOx
            end
            if haskey(EDGAR_SPECIES, edgar_sp)
                emissions[sp.name] = load_emissions(mgr, edgar_sp;
                                                      hgrid=hgrid, gpu=gpu)
            end
        end
    end

    @info "Loaded emissions for: $(join(keys(emissions), ", "))"
    return emissions
end

"""Inject emissions into the lowest model level."""
function _inject_emissions!(state, dt)
    for (sp_name, emi_field) in state.emissions
        sp_idx = species_index(state.mechanism.species, sp_name)
        sp = state.mechanism.species[sp_idx]

        conc_cpu = Array(state.conc)
        emi_cpu = Array(emi_field)
        dp_cpu = Array(layer_thickness_dp(state.vgrid, Array(state.met.ps)))

        nlon, nlat = size(emi_cpu)
        k_sfc = state.vgrid.nlevels

        for j in 1:nlat, i in 1:nlon
            E = emi_cpu[i, j]  # kg m⁻² s⁻¹
            if E <= 0
                continue
            end

            dp_sfc = max(dp_cpu[i, j, k_sfc], 1.0)
            # Layer thickness [m]
            rho_sfc = Array(state.met.ps)[i, j] /
                      (Constants.R_GAS / Constants.M_AIR * 288.0)
            dz = dp_sfc / (rho_sfc * Constants.G)

            # Convert kg m⁻² s⁻¹ → molec cm⁻³ s⁻¹
            dn_dt = E * Constants.N_A / (sp.mw * dz * 1e6)

            conc_cpu[i, j, k_sfc, sp_idx] += dn_dt * dt
        end

        state.conc .= (typeof(state.conc) != typeof(conc_cpu)) ?
            to_device(conc_cpu; gpu=true) : conc_cpu
    end
end

"""Update meteorology from ERA5 data."""
function _update_meteorology!(state, met_manager, t)
    year = Dates.year(t)
    month = Dates.month(t)
    day = Dates.day(t)
    hour = Dates.hour(t)

    try
        state.met = load_era5_fields(met_manager, year, month, day, hour;
                                      vgrid=state.vgrid, gpu=state.config.gpu)
        _compute_M_air!(state.M_air, state.met.T, state.vgrid, state.met.ps)
        @info "Updated meteorology for $t"
    catch e
        @warn "Could not load ERA5 data for $t: $e. Using previous/default fields."
    end
end

"""Compute overhead O3 column [DU] from 3-D O3 concentrations."""
function _compute_o3_column(conc, o3_idx, vgrid, ps)
    conc_cpu = Array(conc)
    ps_cpu = Array(ps)
    nlon, nlat, nlevels = size(conc_cpu, 1), size(conc_cpu, 2), size(conc_cpu, 3)

    o3_col = zeros(Float64, nlon, nlat)
    dp = layer_thickness_dp(vgrid, ps_cpu)

    for j in 1:nlat, i in 1:nlon
        col_molec_cm2 = 0.0
        for k in 1:nlevels
            n_o3 = max(conc_cpu[i, j, k, o3_idx], 0.0)  # molec cm⁻³
            # Layer thickness in cm
            rho = (vgrid.a_full[k] + vgrid.b_full[k] * ps_cpu[i, j]) /
                  (Constants.R_GAS / Constants.M_AIR * 250.0)
            dz_cm = dp[i, j, k] / (rho * Constants.G) * 100.0
            col_molec_cm2 += n_o3 * dz_cm
        end
        # Convert molec cm⁻² to Dobson Units (1 DU = 2.687e16 molec cm⁻²)
        o3_col[i, j] = col_molec_cm2 / 2.687e16
    end

    return o3_col
end

"""Estimate cloud fraction from specific humidity and temperature."""
function _estimate_cloud_fraction(q, T, vgrid, ps)
    q_cpu = Array(q)
    T_cpu = Array(T)
    ps_cpu = Array(ps)
    nlon, nlat, nlevels = size(q_cpu)

    cloud_frac = zeros(Float64, nlon, nlat)

    for j in 1:nlat, i in 1:nlon
        max_rh = 0.0
        for k in 1:nlevels
            T_k = T_cpu[i, j, k]
            p_k = vgrid.a_full[k] + vgrid.b_full[k] * ps_cpu[i, j]
            # Saturation specific humidity
            es = 611.2 * exp(17.67 * (T_k - 273.15) / (T_k - 29.65))
            qs = 0.622 * es / max(p_k - es, 1.0)
            rh = q_cpu[i, j, k] / max(qs, 1e-10)
            max_rh = max(max_rh, rh)
        end
        # Simple threshold: cloud if RH > 0.7
        cloud_frac[i, j] = clamp((max_rh - 0.7) / 0.3, 0.0, 1.0)
    end

    return cloud_frac
end
