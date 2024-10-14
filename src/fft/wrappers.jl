# Key: context (device), fft type (fwd, inv), xdims, x eltype, inplace or not, region.
const HandleCacheKey = Tuple{HIPContext, rocfft_transform_type, Dims, Type, Bool, Any}
# Value: plan, worksize.
const HandleCacheValue = Tuple{rocfft_plan, Int}
const IDLE_HANDLES = HandleCache{HandleCacheKey, HandleCacheValue}()

function get_plan(args...)
    rocfft_setup_once()
    handle, worksize = pop!(IDLE_HANDLES, (AMDGPU.context(), args...)) do
        create_plan(args...)
    end
    workarea = ROCVector{Int8}(undef, worksize)
    return handle, workarea
end

function release_plan!(plan)
    key = (
        AMDGPU.context(), plan.xtype, plan.sz,
        eltype(plan), is_inplace(plan), plan.region)
    value = (plan.handle, length(plan.workarea))
    push!(IDLE_HANDLES, key, value) do
        destroy_plan!(plan)
    end
end

function destroy_plan!(plan)
    rocfft_plan_destroy(plan.handle)
end

#TODO: add bound checks
function create_plan(xtype::rocfft_transform_type, xdims, T, inplace, region)
    precision = (real(T) == Float64) ? rocfft_precision_double : rocfft_precision_single
    placement = inplace ? rocfft_placement_inplace : rocfft_placement_notinplace

    nrank = length(region)
    sz = [xdims[i] for i in region]
    batch = prod(xdims) ÷ prod(sz)

    handle_ref = Ref{rocfft_plan}()
    worksize_ref = Ref{Csize_t}()

    plan_desc_ref = Ref{rocfft_plan_description}()
    rocfft_plan_description_create(plan_desc_ref)
    description = plan_desc_ref[]

    in_array_type = out_array_type = rocfft_array_type_complex_interleaved

    # Calculate strides and distances
    istrides = ones(Int, nrank)
    ostrides = ones(Int, nrank)
    for i in 1:nrank
        istrides[i] = ostrides[i] = prod(xdims[1:region[i]-1])
    end

    idist = odist = prod(xdims[1:maximum(region)])

    rocfft_plan_description_set_data_layout(
        description, in_array_type, out_array_type, C_NULL, C_NULL,
        nrank, istrides, idist,
        nrank, ostrides, odist
    )

    rocfft_plan_create(
        handle_ref, placement, xtype, precision,
        nrank, sz, batch, description
    )

    rocfft_plan_description_destroy(description)

    rocfft_plan_get_work_buffer_size(handle_ref[], worksize_ref)
    return handle_ref[], Int(worksize_ref[])
end