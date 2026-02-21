# ---------------------------------------------------------------------------
# Device-agnostic array utilities (CPU / GPU)
# ---------------------------------------------------------------------------

"""
    device_array(T, dims...; gpu=false)

Allocate an array on the appropriate device. Returns a `CuArray` when
`gpu=true` and CUDA is available, otherwise a plain `Array`.
"""
function device_array(::Type{T}, dims...; gpu::Bool=false) where T
    if gpu && HAS_CUDA
        return CUDA.zeros(T, dims...)
    else
        return zeros(T, dims...)
    end
end

"""
    to_device(arr; gpu=false)

Transfer an existing array to the target device.
"""
function to_device(arr::AbstractArray; gpu::Bool=false)
    if gpu && HAS_CUDA
        return CUDA.CuArray(arr)
    else
        return Array(arr)
    end
end

"""
    synchronize_device(; gpu=false)

Synchronize the GPU stream if running on GPU.
"""
function synchronize_device(; gpu::Bool=false)
    if gpu && HAS_CUDA
        CUDA.synchronize()
    end
end

"""
    array_backend(; gpu=false)

Return the appropriate KernelAbstractions backend.
"""
function array_backend(; gpu::Bool=false)
    if gpu && HAS_CUDA
        return CUDABackend()
    else
        return CPU()
    end
end
