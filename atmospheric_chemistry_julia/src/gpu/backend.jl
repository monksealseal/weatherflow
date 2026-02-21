# ---------------------------------------------------------------------------
# GPU backend — dispatch and management
# ---------------------------------------------------------------------------

"""
    GPUBackend

Manages GPU execution for the atmospheric chemistry model.
Provides methods to launch kernels and manage device memory.
"""
struct GPUBackend
    enabled :: Bool
    device_id :: Int
    backend :: Any  # KernelAbstractions backend
end

function GPUBackend(; gpu::Bool=false, device_id::Int=0)
    if gpu && HAS_CUDA
        if device_id < CUDA.ndevices()
            CUDA.device!(device_id)
        end
        @info "GPU backend enabled: $(CUDA.name(CUDA.device()))"
        return GPUBackend(true, device_id, CUDABackend())
    else
        if gpu
            @warn "GPU requested but CUDA not available. Using CPU."
        end
        return GPUBackend(false, 0, CPU())
    end
end

"""
    launch_advection!(backend, conc, flux_x, flux_y, flux_z, dx, dy, dp, dt)

Launch the GPU advection kernel.
"""
function launch_advection!(gpub::GPUBackend, conc, flux_x, flux_y, flux_z,
                            dx, dy, dp, dt)
    nlon, nlat, nlevels = size(conc)[1:3]
    kernel! = advection_kernel!(gpub.backend, 256)
    kernel!(conc, flux_x, flux_y, flux_z, dx, dy, dp, dt,
            nlon, nlat, nlevels; ndrange=(nlon, nlat, nlevels))
    synchronize_device(; gpu=gpub.enabled)
end

"""
    launch_photolysis!(backend, j_rates, lat, lon, p, ps, o3_col, cloud_frac, dt)

Launch the GPU photolysis kernel.
"""
function launch_photolysis!(gpub::GPUBackend, j_rates, lat, lon, p, ps,
                             o3_col, cloud_frac, datetime::DateTime)
    nlon = length(lon)
    nlat = length(lat)
    nlevels = size(p, 3)

    hour_utc = Float64(Dates.hour(datetime) + Dates.minute(datetime) / 60)
    doy = Dates.dayofyear(datetime)

    kernel! = photolysis_kernel!(gpub.backend, 256)
    kernel!(j_rates, lat, lon, p, ps, o3_col, cloud_frac,
            hour_utc, doy, nlon, nlat, nlevels;
            ndrange=(nlon, nlat, nlevels))
    synchronize_device(; gpu=gpub.enabled)
end

"""
    launch_emissions!(backend, conc, emission_rate, dp, species_idx, dt, mw)

Launch the GPU emission injection kernel.
"""
function launch_emissions!(gpub::GPUBackend, conc, emission_rate, dp,
                            species_idx::Int, dt::Float64, mw::Float64)
    nlon, nlat = size(emission_rate)
    kernel! = emission_kernel!(gpub.backend, 256)
    kernel!(conc, emission_rate, dp, species_idx, dt, mw,
            nlon, nlat; ndrange=(nlon, nlat))
    synchronize_device(; gpu=gpub.enabled)
end

"""
    launch_deposition!(backend, conc, v_dep, dp_sfc, ps, dt)

Launch the GPU deposition kernel.
"""
function launch_deposition!(gpub::GPUBackend, conc, v_dep, dp_sfc, ps, dt)
    nlon, nlat, _, nspec = size(conc)
    kernel! = deposition_kernel!(gpub.backend, 256)
    kernel!(conc, v_dep, dp_sfc, ps, dt, nlon, nlat, nspec;
            ndrange=(nlon, nlat, nspec))
    synchronize_device(; gpu=gpub.enabled)
end

"""
    gpu_memory_info()

Print GPU memory usage information.
"""
function gpu_memory_info()
    if HAS_CUDA
        free = CUDA.available_memory()
        total = CUDA.total_memory()
        used = total - free
        @info @sprintf("GPU Memory: %.1f / %.1f GB (%.1f%% used)",
                       used / 1e9, total / 1e9, 100 * used / total)
    else
        @info "No GPU available"
    end
end
