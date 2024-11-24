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

# Helper function to calculate max dimensions like in CUDA.jl
function plan_max_dims(region, sz)
    if (region[1] == 1 && (length(region) <= 1 || all(diff(collect(region)) .== 1)))
        return length(sz)
    else
        return region[end]
    end
end

# Helper function to get view of front dimensions
function front_view(X, md)
    t = ntuple((d)->ifelse(d<=md, Colon(), 1), ndims(X))
    @view X[t...]
end

function create_plan(xtype::rocfft_transform_type, xdims, T, inplace, region)
    precision = (real(T) == Float64) ?
        rocfft_precision_double : rocfft_precision_single
    placement = inplace ?
        rocfft_placement_inplace : rocfft_placement_notinplace

    region_tuple = Tuple(region)
    
    if length(region_tuple) > 1
        if any(diff(collect(region_tuple)) .< 1)
            throw(ArgumentError("region must be an increasing sequence"))
        end
    end
    if any(region_tuple .< 1 .|| region_tuple .> length(xdims))
        throw(ArgumentError("region can only refer to valid dimensions"))
    end

    nrank = length(region_tuple)
    sz = [xdims[i] for i in region_tuple]
    csz = copy(sz)
    csz[1] = div(sz[1], 2) + 1
    batch = prod(xdims) ÷ prod(sz)

    worksize_ref = Ref{Csize_t}()
    rsz = length(sz) > 1 ? reverse(sz) : sz
    if nrank > 3
        throw(ArgumentError("only up to three transform dimensions are allowed in one plan"))
    end

    handle_ref = Ref{rocfft_plan}()
    
    if batch == 1
        rocfft_plan_create(
            handle_ref, placement, xtype, precision, nrank, rsz, 1, C_NULL)
    else
        plan_desc_ref = Ref{rocfft_plan_description}()
        rocfft_plan_description_create(plan_desc_ref)
        description = plan_desc_ref[]

        try
            if xtype == rocfft_transform_type_real_forward
                in_array_type = rocfft_array_type_real
                out_array_type = rocfft_array_type_hermitian_interleaved
            elseif xtype == rocfft_transform_type_real_inverse
                in_array_type = rocfft_array_type_hermitian_interleaved
                out_array_type = rocfft_array_type_real
            else
                in_array_type = rocfft_array_type_complex_interleaved
                out_array_type = rocfft_array_type_complex_interleaved
            end

            if ((region_tuple...,) == ((1:nrank)...,))
                rocfft_plan_description_set_data_layout(
                    description, in_array_type, out_array_type,
                    C_NULL, C_NULL,
                    nrank, C_NULL, 0,
                    nrank, C_NULL, 0)
                
                rocfft_plan_create(
                    handle_ref, placement, xtype, precision,
                    nrank, rsz, batch, description)
            else
                if region_tuple[1] == 1  # First dimension
                    istrides = [1]
                    idist = prod(sz)
                    ostrides = istrides
                    odist = idist

                    rocfft_plan_description_set_data_layout(
                        description, in_array_type, out_array_type,
                        C_NULL, C_NULL,
                        length(istrides), istrides, idist,
                        length(ostrides), ostrides, odist)
                else  # Other dimensions
                    if nrank == 1 || all(diff(collect(region_tuple)) .== 1)
                        istride = prod(xdims[1:region_tuple[1]-1])
                        idist = 1
                        cdist = 1
                        
                        inembed = rsz
                        cnembed = (length(csz) > 1) ? reverse(csz) : [csz[1]]
                        ostride = istride
                        odist = idist
                        onembed = inembed
                        
                        if xtype == rocfft_transform_type_real_forward
                            odist = cdist
                            onembed = cnembed
                        elseif xtype == rocfft_transform_type_real_inverse
                            idist = cdist
                            inembed = cnembed
                        end
                        
                        rocfft_plan_description_set_data_layout(
                            description, in_array_type, out_array_type,
                            C_NULL, C_NULL,
                            1, [istride], idist,
                            1, [ostride], odist)
                    else
                        istride = prod(xdims[1:region_tuple[1]-1])
                        idist = 1
                        cdist = 1
                        
                        inembed = reverse(rsz)
                        onembed = inembed
                        ostride = istride
                        odist = idist

                        rocfft_plan_description_set_data_layout(
                            description, in_array_type, out_array_type,
                            C_NULL, C_NULL,
                            1, [istride], idist,
                            1, [ostride], odist)
                    end
                end

                rocfft_plan_create(
                    handle_ref, placement, xtype, precision,
                    nrank, rsz, batch, description)
            end

            rocfft_plan_get_work_buffer_size(handle_ref[], worksize_ref)
            return handle_ref[], Int(worksize_ref[])
        finally
            rocfft_plan_description_destroy(description)
        end
    end
end