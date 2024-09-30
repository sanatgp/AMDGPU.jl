using CEnum

mutable struct _rocblas_handle end

const rocblas_handle = Ptr{_rocblas_handle}

@cenum rocblas_status_::UInt32 begin
    rocblas_status_success = 0
    rocblas_status_invalid_handle = 1
    rocblas_status_not_implemented = 2
    rocblas_status_invalid_pointer = 3
    rocblas_status_invalid_size = 4
    rocblas_status_memory_error = 5
    rocblas_status_internal_error = 6
    rocblas_status_perf_degraded = 7
    rocblas_status_size_query_mismatch = 8
    rocblas_status_size_increased = 9
    rocblas_status_size_unchanged = 10
    rocblas_status_invalid_value = 11
    rocblas_status_continue = 12
    rocblas_status_check_numerics_fail = 13
    rocblas_status_excluded_from_build = 14
    rocblas_status_arch_mismatch = 15
end

const rocblas_status = rocblas_status_

function rocblas_set_start_stop_events(handle, startEvent, stopEvent)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_start_stop_events(handle::rocblas_handle,
                                                    startEvent::hipEvent_t,
                                                    stopEvent::hipEvent_t)::rocblas_status
end

struct rocblas_bfloat16
    data::UInt16
end

struct rocblas_f8
    data::UInt8
end

struct rocblas_bf8
    data::UInt8
end

mutable struct rocblas_device_malloc_base end

const rocblas_int = Int32

const rocblas_stride = Int64

const rocblas_float = Cfloat

const rocblas_double = Cdouble

const rocblas_half = Float16

const rocblas_float_complex = ComplexF32

const rocblas_double_complex = ComplexF64

@cenum rocblas_operation_::UInt32 begin
    rocblas_operation_none = 111
    rocblas_operation_transpose = 112
    rocblas_operation_conjugate_transpose = 113
end

const rocblas_operation = rocblas_operation_

@cenum rocblas_fill_::UInt32 begin
    rocblas_fill_upper = 121
    rocblas_fill_lower = 122
    rocblas_fill_full = 123
end

const rocblas_fill = rocblas_fill_

@cenum rocblas_diagonal_::UInt32 begin
    rocblas_diagonal_non_unit = 131
    rocblas_diagonal_unit = 132
end

const rocblas_diagonal = rocblas_diagonal_

@cenum rocblas_side_::UInt32 begin
    rocblas_side_left = 141
    rocblas_side_right = 142
    rocblas_side_both = 143
end

const rocblas_side = rocblas_side_

@cenum rocblas_datatype_::UInt32 begin
    rocblas_datatype_f16_r = 150
    rocblas_datatype_f32_r = 151
    rocblas_datatype_f64_r = 152
    rocblas_datatype_f16_c = 153
    rocblas_datatype_f32_c = 154
    rocblas_datatype_f64_c = 155
    rocblas_datatype_i8_r = 160
    rocblas_datatype_u8_r = 161
    rocblas_datatype_i32_r = 162
    rocblas_datatype_u32_r = 163
    rocblas_datatype_i8_c = 164
    rocblas_datatype_u8_c = 165
    rocblas_datatype_i32_c = 166
    rocblas_datatype_u32_c = 167
    rocblas_datatype_bf16_r = 168
    rocblas_datatype_bf16_c = 169
    rocblas_datatype_f8_r = 170
    rocblas_datatype_bf8_r = 171
    rocblas_datatype_invalid = 255
end

const rocblas_datatype = rocblas_datatype_

@cenum rocblas_computetype_::UInt32 begin
    rocblas_compute_type_f32 = 300
    rocblas_compute_type_f8_f8_f32 = 301
    rocblas_compute_type_f8_bf8_f32 = 302
    rocblas_compute_type_bf8_f8_f32 = 303
    rocblas_compute_type_bf8_bf8_f32 = 304
    rocblas_compute_type_invalid = 455
end

const rocblas_computetype = rocblas_computetype_

@cenum rocblas_pointer_mode_::UInt32 begin
    rocblas_pointer_mode_host = 0
    rocblas_pointer_mode_device = 1
end

const rocblas_pointer_mode = rocblas_pointer_mode_

@cenum rocblas_atomics_mode_::UInt32 begin
    rocblas_atomics_not_allowed = 0
    rocblas_atomics_allowed = 1
end

const rocblas_atomics_mode = rocblas_atomics_mode_

@cenum rocblas_performance_metric_::UInt32 begin
    rocblas_default_performance_metric = 0
    rocblas_device_efficiency_performance_metric = 1
    rocblas_cu_efficiency_performance_metric = 2
end

const rocblas_performance_metric = rocblas_performance_metric_

@cenum rocblas_layer_mode_::UInt32 begin
    rocblas_layer_mode_none = 0
    rocblas_layer_mode_log_trace = 1
    rocblas_layer_mode_log_bench = 2
    rocblas_layer_mode_log_profile = 4
end

const rocblas_layer_mode = rocblas_layer_mode_

@cenum rocblas_gemm_algo_::UInt32 begin
    rocblas_gemm_algo_standard = 0
    rocblas_gemm_algo_solution_index = 1
end

const rocblas_gemm_algo = rocblas_gemm_algo_

@cenum rocblas_geam_ex_operation_::UInt32 begin
    rocblas_geam_ex_operation_min_plus = 0
    rocblas_geam_ex_operation_plus_min = 1
end

const rocblas_geam_ex_operation = rocblas_geam_ex_operation_

@cenum rocblas_gemm_flags_::UInt32 begin
    rocblas_gemm_flags_none = 0
    rocblas_gemm_flags_use_cu_efficiency = 2
    rocblas_gemm_flags_fp16_alt_impl = 4
    rocblas_gemm_flags_check_solution_index = 8
    rocblas_gemm_flags_fp16_alt_impl_rnz = 16
    rocblas_gemm_flags_stochastic_rounding = 32
end

const rocblas_gemm_flags = rocblas_gemm_flags_

struct rocblas_union_u
    data::NTuple{16,UInt8}
end

function Base.getproperty(x::Ptr{rocblas_union_u}, f::Symbol)
    f === :h && return Ptr{rocblas_half}(x + 0)
    f === :s && return Ptr{Cfloat}(x + 0)
    f === :d && return Ptr{Cdouble}(x + 0)
    f === :i && return Ptr{Int32}(x + 0)
    f === :c && return Ptr{rocblas_float_complex}(x + 0)
    f === :z && return Ptr{rocblas_double_complex}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::rocblas_union_u, f::Symbol)
    r = Ref{rocblas_union_u}(x)
    ptr = Base.unsafe_convert(Ptr{rocblas_union_u}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{rocblas_union_u}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const rocblas_union_t = rocblas_union_u

@cenum rocblas_check_numerics_mode_::UInt32 begin
    rocblas_check_numerics_mode_no_check = 0
    rocblas_check_numerics_mode_info = 1
    rocblas_check_numerics_mode_warn = 2
    rocblas_check_numerics_mode_fail = 4
end

const rocblas_check_numerics_mode = rocblas_check_numerics_mode_

@cenum rocblas_math_mode_::UInt32 begin
    rocblas_default_math = 0
    rocblas_xf32_xdl_math_op = 1
end

const rocblas_math_mode = rocblas_math_mode_

function rocblas_create_handle(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_create_handle(handle::Ptr{rocblas_handle})::rocblas_status
end

function rocblas_destroy_handle(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_destroy_handle(handle::rocblas_handle)::rocblas_status
end

function rocblas_set_stream(handle, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_stream(handle::rocblas_handle,
                                         stream::hipStream_t)::rocblas_status
end

function rocblas_get_stream(handle, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_stream(handle::rocblas_handle,
                                         stream::Ptr{hipStream_t})::rocblas_status
end

function rocblas_set_pointer_mode(handle, pointer_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_pointer_mode(handle::rocblas_handle,
                                               pointer_mode::rocblas_pointer_mode)::rocblas_status
end

function rocblas_get_pointer_mode(handle, pointer_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_pointer_mode(handle::rocblas_handle,
                                               pointer_mode::Ptr{rocblas_pointer_mode})::rocblas_status
end

function rocblas_set_atomics_mode(handle, atomics_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_atomics_mode(handle::rocblas_handle,
                                               atomics_mode::rocblas_atomics_mode)::rocblas_status
end

function rocblas_get_atomics_mode(handle, atomics_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_atomics_mode(handle::rocblas_handle,
                                               atomics_mode::Ptr{rocblas_atomics_mode})::rocblas_status
end

function rocblas_set_math_mode(handle, math_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_math_mode(handle::rocblas_handle,
                                            math_mode::rocblas_math_mode)::rocblas_status
end

function rocblas_get_math_mode(handle, math_mode)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_math_mode(handle::rocblas_handle,
                                            math_mode::Ptr{rocblas_math_mode})::rocblas_status
end

function rocblas_pointer_to_mode(ptr)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_pointer_to_mode(ptr::Ptr{Cvoid})::rocblas_pointer_mode
end

function rocblas_set_vector(n, elem_size, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_vector(n::rocblas_int, elem_size::rocblas_int,
                                         x::Ptr{Cvoid}, incx::rocblas_int, y::Ptr{Cvoid},
                                         incy::rocblas_int)::rocblas_status
end

function rocblas_get_vector(n, elem_size, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_vector(n::rocblas_int, elem_size::rocblas_int,
                                         x::Ptr{Cvoid}, incx::rocblas_int, y::Ptr{Cvoid},
                                         incy::rocblas_int)::rocblas_status
end

function rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_matrix(rows::rocblas_int, cols::rocblas_int,
                                         elem_size::rocblas_int, a::Ptr{Cvoid},
                                         lda::rocblas_int, b::Ptr{Cvoid},
                                         ldb::rocblas_int)::rocblas_status
end

function rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_matrix(rows::rocblas_int, cols::rocblas_int,
                                         elem_size::rocblas_int, a::Ptr{Cvoid},
                                         lda::rocblas_int, b::Ptr{Cvoid},
                                         ldb::rocblas_int)::rocblas_status
end

function rocblas_set_vector_async(n, elem_size, x, incx, y, incy, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_vector_async(n::rocblas_int, elem_size::rocblas_int,
                                               x::Ptr{Cvoid}, incx::rocblas_int,
                                               y::Ptr{Cvoid}, incy::rocblas_int,
                                               stream::hipStream_t)::rocblas_status
end

function rocblas_get_vector_async(n, elem_size, x, incx, y, incy, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_vector_async(n::rocblas_int, elem_size::rocblas_int,
                                               x::Ptr{Cvoid}, incx::rocblas_int,
                                               y::Ptr{Cvoid}, incy::rocblas_int,
                                               stream::hipStream_t)::rocblas_status
end

function rocblas_set_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_matrix_async(rows::rocblas_int, cols::rocblas_int,
                                               elem_size::rocblas_int, a::Ptr{Cvoid},
                                               lda::rocblas_int, b::Ptr{Cvoid},
                                               ldb::rocblas_int,
                                               stream::hipStream_t)::rocblas_status
end

function rocblas_get_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_matrix_async(rows::rocblas_int, cols::rocblas_int,
                                               elem_size::rocblas_int, a::Ptr{Cvoid},
                                               lda::rocblas_int, b::Ptr{Cvoid},
                                               ldb::rocblas_int,
                                               stream::hipStream_t)::rocblas_status
end

function rocblas_set_solution_fitness_query(handle, fitness)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_solution_fitness_query(handle::rocblas_handle,
                                                         fitness::Ptr{Cdouble})::rocblas_status
end

function rocblas_set_performance_metric(handle, metric)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_performance_metric(handle::rocblas_handle,
                                                     metric::rocblas_performance_metric)::rocblas_status
end

function rocblas_get_performance_metric(handle, metric)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_performance_metric(handle::rocblas_handle,
                                                     metric::Ptr{rocblas_performance_metric})::rocblas_status
end

function rocblas_sscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sscal(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_dscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dscal(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_cscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cscal(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_zscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zscal(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_csscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csscal(handle::rocblas_handle, n::rocblas_int,
                                     alpha::Ptr{Cfloat}, x::Ptr{rocblas_float_complex},
                                     incx::rocblas_int)::rocblas_status
end

function rocblas_zdscal(handle, n, alpha, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdscal(handle::rocblas_handle, n::rocblas_int,
                                     alpha::Ptr{Cdouble}, x::Ptr{rocblas_double_complex},
                                     incx::rocblas_int)::rocblas_status
end

function rocblas_sscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sscal_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{Cfloat}, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dscal_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{Cdouble}, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cscal_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zscal_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_csscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csscal_batched(handle::rocblas_handle, n::rocblas_int,
                                             alpha::Ptr{Cfloat},
                                             x::Ptr{Ptr{rocblas_float_complex}},
                                             incx::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdscal_batched(handle, n, alpha, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdscal_batched(handle::rocblas_handle, n::rocblas_int,
                                             alpha::Ptr{Cdouble},
                                             x::Ptr{Ptr{rocblas_double_complex}},
                                             incx::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_sscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_csscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     alpha::Ptr{Cfloat},
                                                     x::Ptr{rocblas_float_complex},
                                                     incx::rocblas_int,
                                                     stride_x::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdscal_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     alpha::Ptr{Cdouble},
                                                     x::Ptr{rocblas_double_complex},
                                                     incx::rocblas_int,
                                                     stride_x::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_scopy(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scopy(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dcopy(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dcopy(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_ccopy(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ccopy(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zcopy(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zcopy(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_scopy_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scopy_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dcopy_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dcopy_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ccopy_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ccopy_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zcopy_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zcopy_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_scopy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scopy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stridex::rocblas_stride, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dcopy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dcopy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ccopy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ccopy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zcopy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zcopy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sdot(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdot(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                   incx::rocblas_int, y::Ptr{Cfloat}, incy::rocblas_int,
                                   result::Ptr{Cfloat})::rocblas_status
end

function rocblas_ddot(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddot(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                   incx::rocblas_int, y::Ptr{Cdouble}, incy::rocblas_int,
                                   result::Ptr{Cdouble})::rocblas_status
end

function rocblas_hdot(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hdot(handle::rocblas_handle, n::rocblas_int,
                                   x::Ptr{rocblas_half}, incx::rocblas_int,
                                   y::Ptr{rocblas_half}, incy::rocblas_int,
                                   result::Ptr{rocblas_half})::rocblas_status
end

function rocblas_bfdot(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_bfdot(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_bfloat16}, incx::rocblas_int,
                                    y::Ptr{rocblas_bfloat16}, incy::rocblas_int,
                                    result::Ptr{rocblas_bfloat16})::rocblas_status
end

function rocblas_cdotu(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotu(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotu(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotu(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_cdotc(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotc(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotc(handle, n, x, incx, y, incy, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotc(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_sdot_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                           batch_count::rocblas_int,
                                           result::Ptr{Cfloat})::rocblas_status
end

function rocblas_ddot_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                           batch_count::rocblas_int,
                                           result::Ptr{Cdouble})::rocblas_status
end

function rocblas_hdot_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hdot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{rocblas_half}}, incx::rocblas_int,
                                           y::Ptr{Ptr{rocblas_half}}, incy::rocblas_int,
                                           batch_count::rocblas_int,
                                           result::Ptr{rocblas_half})::rocblas_status
end

function rocblas_bfdot_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_bfdot_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_bfloat16}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_bfloat16}},
                                            incy::rocblas_int, batch_count::rocblas_int,
                                            result::Ptr{rocblas_bfloat16})::rocblas_status
end

function rocblas_cdotu_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotu_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int, batch_count::rocblas_int,
                                            result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotu_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotu_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int, batch_count::rocblas_int,
                                            result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_cdotc_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotc_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int, batch_count::rocblas_int,
                                            result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotc_batched(handle, n, x, incx, y, incy, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotc_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int, batch_count::rocblas_int,
                                            result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_sdot_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                      batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{Cfloat}, incx::rocblas_int,
                                                   stridex::rocblas_stride, y::Ptr{Cfloat},
                                                   incy::rocblas_int,
                                                   stridey::rocblas_stride,
                                                   batch_count::rocblas_int,
                                                   result::Ptr{Cfloat})::rocblas_status
end

function rocblas_ddot_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                      batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{Cdouble}, incx::rocblas_int,
                                                   stridex::rocblas_stride, y::Ptr{Cdouble},
                                                   incy::rocblas_int,
                                                   stridey::rocblas_stride,
                                                   batch_count::rocblas_int,
                                                   result::Ptr{Cdouble})::rocblas_status
end

function rocblas_hdot_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                      batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hdot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{rocblas_half}, incx::rocblas_int,
                                                   stridex::rocblas_stride,
                                                   y::Ptr{rocblas_half}, incy::rocblas_int,
                                                   stridey::rocblas_stride,
                                                   batch_count::rocblas_int,
                                                   result::Ptr{rocblas_half})::rocblas_status
end

function rocblas_bfdot_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_bfdot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_bfloat16},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_bfloat16},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    result::Ptr{rocblas_bfloat16})::rocblas_status
end

function rocblas_cdotu_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotu_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotu_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotu_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_cdotc_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdotc_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    result::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zdotc_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdotc_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    result::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_sswap(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sswap(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dswap(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dswap(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_cswap(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cswap(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zswap(handle, n, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zswap(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_sswap_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sswap_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dswap_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dswap_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cswap_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cswap_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zswap_batched(handle, n, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zswap_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sswap_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stridex::rocblas_stride, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dswap_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cswap_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zswap_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_saxpy(handle, n, alpha, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_saxpy(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::rocblas_int,
                                    y::Ptr{Cfloat}, incy::rocblas_int)::rocblas_status
end

function rocblas_daxpy(handle, n, alpha, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_daxpy(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::rocblas_int,
                                    y::Ptr{Cdouble}, incy::rocblas_int)::rocblas_status
end

function rocblas_haxpy(handle, n, alpha, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_haxpy(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{rocblas_half}, x::Ptr{rocblas_half},
                                    incx::rocblas_int, y::Ptr{rocblas_half},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_caxpy(handle, n, alpha, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_caxpy(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zaxpy(handle, n, alpha, x, incx, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zaxpy(handle::rocblas_handle, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_haxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_haxpy_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{rocblas_half},
                                            x::Ptr{Ptr{rocblas_half}}, incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_half}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_saxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_saxpy_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{Cfloat}, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int, y::Ptr{Ptr{Cfloat}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_daxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_daxpy_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{Cdouble}, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int, y::Ptr{Ptr{Cdouble}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_caxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_caxpy_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zaxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zaxpy_batched(handle::rocblas_handle, n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_haxpy_strided_batched(handle, n, alpha, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_haxpy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{rocblas_half},
                                                    x::Ptr{rocblas_half}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_half}, incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_saxpy_strided_batched(handle, n, alpha, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_saxpy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_daxpy_strided_batched(handle, n, alpha, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_daxpy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_caxpy_strided_batched(handle, n, alpha, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_caxpy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zaxpy_strided_batched(handle, n, alpha, x, incx, stridex, y, incy, stridey,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zaxpy_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sasum(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sasum(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, result::Ptr{Cfloat})::rocblas_status
end

function rocblas_dasum(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dasum(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, result::Ptr{Cdouble})::rocblas_status
end

function rocblas_scasum(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scasum(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                     result::Ptr{Cfloat})::rocblas_status
end

function rocblas_dzasum(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dzasum(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                     result::Ptr{Cdouble})::rocblas_status
end

function rocblas_sasum_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sasum_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            batch_count::rocblas_int,
                                            results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dasum_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dasum_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            batch_count::rocblas_int,
                                            results::Ptr{Cdouble})::rocblas_status
end

function rocblas_scasum_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scasum_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_float_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dzasum_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dzasum_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_double_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             results::Ptr{Cdouble})::rocblas_status
end

function rocblas_sasum_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sasum_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dasum_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dasum_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    results::Ptr{Cdouble})::rocblas_status
end

function rocblas_scasum_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scasum_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_float_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dzasum_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dzasum_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_double_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     results::Ptr{Cdouble})::rocblas_status
end

function rocblas_snrm2(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_snrm2(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, result::Ptr{Cfloat})::rocblas_status
end

function rocblas_dnrm2(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dnrm2(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, result::Ptr{Cdouble})::rocblas_status
end

function rocblas_scnrm2(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scnrm2(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                     result::Ptr{Cfloat})::rocblas_status
end

function rocblas_dznrm2(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dznrm2(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                     result::Ptr{Cdouble})::rocblas_status
end

function rocblas_snrm2_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_snrm2_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            batch_count::rocblas_int,
                                            results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dnrm2_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dnrm2_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            batch_count::rocblas_int,
                                            results::Ptr{Cdouble})::rocblas_status
end

function rocblas_scnrm2_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scnrm2_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_float_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dznrm2_batched(handle, n, x, incx, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dznrm2_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_double_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             results::Ptr{Cdouble})::rocblas_status
end

function rocblas_snrm2_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_snrm2_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dnrm2_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dnrm2_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    batch_count::rocblas_int,
                                                    results::Ptr{Cdouble})::rocblas_status
end

function rocblas_scnrm2_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scnrm2_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_float_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     results::Ptr{Cfloat})::rocblas_status
end

function rocblas_dznrm2_strided_batched(handle, n, x, incx, stridex, batch_count, results)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dznrm2_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_double_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     results::Ptr{Cdouble})::rocblas_status
end

function rocblas_isamax(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamax(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                     incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamax(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamax(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{Cdouble}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamax(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamax(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamax(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamax(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_isamax_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamax_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                             batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamax_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamax_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                             batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamax_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamax_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_float_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamax_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamax_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_double_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_isamax_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamax_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cfloat}, incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamax_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamax_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cdouble}, incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamax_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamax_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_float_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamax_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamax_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_double_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_isamin(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamin(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                     incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamin(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamin(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{Cdouble}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamin(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamin(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamin(handle, n, x, incx, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamin(handle::rocblas_handle, n::rocblas_int,
                                     x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_isamin_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamin_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                             batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamin_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamin_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                             batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamin_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamin_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_float_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamin_batched(handle, n, x, incx, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamin_batched(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Ptr{rocblas_double_complex}},
                                             incx::rocblas_int, batch_count::rocblas_int,
                                             result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_isamin_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_isamin_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cfloat}, incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_idamin_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_idamin_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cdouble}, incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_icamin_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_icamin_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_float_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_izamin_strided_batched(handle, n, x, incx, stridex, batch_count, result)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_izamin_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{rocblas_double_complex},
                                                     incx::rocblas_int,
                                                     stridex::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{rocblas_int})::rocblas_status
end

function rocblas_srot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srot(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                   incx::rocblas_int, y::Ptr{Cfloat}, incy::rocblas_int,
                                   c::Ptr{Cfloat}, s::Ptr{Cfloat})::rocblas_status
end

function rocblas_drot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drot(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                   incx::rocblas_int, y::Ptr{Cdouble}, incy::rocblas_int,
                                   c::Ptr{Cdouble}, s::Ptr{Cdouble})::rocblas_status
end

function rocblas_crot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crot(handle::rocblas_handle, n::rocblas_int,
                                   x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                   y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                   c::Ptr{Cfloat},
                                   s::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_csrot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csrot(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    c::Ptr{Cfloat}, s::Ptr{Cfloat})::rocblas_status
end

function rocblas_zrot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrot(handle::rocblas_handle, n::rocblas_int,
                                   x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                   y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                   c::Ptr{Cdouble},
                                   s::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_zdrot(handle, n, x, incx, y, incy, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdrot(handle::rocblas_handle, n::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    c::Ptr{Cdouble}, s::Ptr{Cdouble})::rocblas_status
end

function rocblas_srot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                           c::Ptr{Cfloat}, s::Ptr{Cfloat},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_drot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                           c::Ptr{Cdouble}, s::Ptr{Cdouble},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_crot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{rocblas_float_complex}},
                                           incx::rocblas_int,
                                           y::Ptr{Ptr{rocblas_float_complex}},
                                           incy::rocblas_int, c::Ptr{Cfloat},
                                           s::Ptr{rocblas_float_complex},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_csrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csrot_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int, c::Ptr{Cfloat},
                                            s::Ptr{Cfloat},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrot_batched(handle::rocblas_handle, n::rocblas_int,
                                           x::Ptr{Ptr{rocblas_double_complex}},
                                           incx::rocblas_int,
                                           y::Ptr{Ptr{rocblas_double_complex}},
                                           incy::rocblas_int, c::Ptr{Cdouble},
                                           s::Ptr{rocblas_double_complex},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdrot_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int, c::Ptr{Cdouble},
                                            s::Ptr{Cdouble},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_srot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s,
                                      batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{Cfloat}, incx::rocblas_int,
                                                   stride_x::rocblas_stride, y::Ptr{Cfloat},
                                                   incy::rocblas_int,
                                                   stride_y::rocblas_stride, c::Ptr{Cfloat},
                                                   s::Ptr{Cfloat},
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_drot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s,
                                      batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{Cdouble}, incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   y::Ptr{Cdouble}, incy::rocblas_int,
                                                   stride_y::rocblas_stride,
                                                   c::Ptr{Cdouble}, s::Ptr{Cdouble},
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_crot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s,
                                      batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{rocblas_float_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   y::Ptr{rocblas_float_complex},
                                                   incy::rocblas_int,
                                                   stride_y::rocblas_stride, c::Ptr{Cfloat},
                                                   s::Ptr{rocblas_float_complex},
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_csrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c,
                                       s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csrot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    c::Ptr{Cfloat}, s::Ptr{Cfloat},
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s,
                                      batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                   x::Ptr{rocblas_double_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   y::Ptr{rocblas_double_complex},
                                                   incy::rocblas_int,
                                                   stride_y::rocblas_stride,
                                                   c::Ptr{Cdouble},
                                                   s::Ptr{rocblas_double_complex},
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c,
                                       s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdrot_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    c::Ptr{Cdouble}, s::Ptr{Cdouble},
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotg(handle, a, b, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotg(handle::rocblas_handle, a::Ptr{Cfloat}, b::Ptr{Cfloat},
                                    c::Ptr{Cfloat}, s::Ptr{Cfloat})::rocblas_status
end

function rocblas_drotg(handle, a, b, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotg(handle::rocblas_handle, a::Ptr{Cdouble},
                                    b::Ptr{Cdouble}, c::Ptr{Cdouble},
                                    s::Ptr{Cdouble})::rocblas_status
end

function rocblas_crotg(handle, a, b, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crotg(handle::rocblas_handle, a::Ptr{rocblas_float_complex},
                                    b::Ptr{rocblas_float_complex}, c::Ptr{Cfloat},
                                    s::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zrotg(handle, a, b, c, s)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrotg(handle::rocblas_handle, a::Ptr{rocblas_double_complex},
                                    b::Ptr{rocblas_double_complex}, c::Ptr{Cdouble},
                                    s::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_srotg_batched(handle, a, b, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotg_batched(handle::rocblas_handle, a::Ptr{Ptr{Cfloat}},
                                            b::Ptr{Ptr{Cfloat}}, c::Ptr{Ptr{Cfloat}},
                                            s::Ptr{Ptr{Cfloat}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotg_batched(handle, a, b, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotg_batched(handle::rocblas_handle, a::Ptr{Ptr{Cdouble}},
                                            b::Ptr{Ptr{Cdouble}}, c::Ptr{Ptr{Cdouble}},
                                            s::Ptr{Ptr{Cdouble}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_crotg_batched(handle, a, b, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crotg_batched(handle::rocblas_handle,
                                            a::Ptr{Ptr{rocblas_float_complex}},
                                            b::Ptr{Ptr{rocblas_float_complex}},
                                            c::Ptr{Ptr{Cfloat}},
                                            s::Ptr{Ptr{rocblas_float_complex}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zrotg_batched(handle, a, b, c, s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrotg_batched(handle::rocblas_handle,
                                            a::Ptr{Ptr{rocblas_double_complex}},
                                            b::Ptr{Ptr{rocblas_double_complex}},
                                            c::Ptr{Ptr{Cdouble}},
                                            s::Ptr{Ptr{rocblas_double_complex}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s,
                                       stride_s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotg_strided_batched(handle::rocblas_handle, a::Ptr{Cfloat},
                                                    stride_a::rocblas_stride,
                                                    b::Ptr{Cfloat},
                                                    stride_b::rocblas_stride,
                                                    c::Ptr{Cfloat},
                                                    stride_c::rocblas_stride,
                                                    s::Ptr{Cfloat},
                                                    stride_s::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s,
                                       stride_s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotg_strided_batched(handle::rocblas_handle, a::Ptr{Cdouble},
                                                    stride_a::rocblas_stride,
                                                    b::Ptr{Cdouble},
                                                    stride_b::rocblas_stride,
                                                    c::Ptr{Cdouble},
                                                    stride_c::rocblas_stride,
                                                    s::Ptr{Cdouble},
                                                    stride_s::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_crotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s,
                                       stride_s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_crotg_strided_batched(handle::rocblas_handle,
                                                    a::Ptr{rocblas_float_complex},
                                                    stride_a::rocblas_stride,
                                                    b::Ptr{rocblas_float_complex},
                                                    stride_b::rocblas_stride,
                                                    c::Ptr{Cfloat},
                                                    stride_c::rocblas_stride,
                                                    s::Ptr{rocblas_float_complex},
                                                    stride_s::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zrotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s,
                                       stride_s, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zrotg_strided_batched(handle::rocblas_handle,
                                                    a::Ptr{rocblas_double_complex},
                                                    stride_a::rocblas_stride,
                                                    b::Ptr{rocblas_double_complex},
                                                    stride_b::rocblas_stride,
                                                    c::Ptr{Cdouble},
                                                    stride_c::rocblas_stride,
                                                    s::Ptr{rocblas_double_complex},
                                                    stride_s::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotm(handle, n, x, incx, y, incy, param)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotm(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, y::Ptr{Cfloat}, incy::rocblas_int,
                                    param::Ptr{Cfloat})::rocblas_status
end

function rocblas_drotm(handle, n, x, incx, y, incy, param)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotm(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, y::Ptr{Cdouble}, incy::rocblas_int,
                                    param::Ptr{Cdouble})::rocblas_status
end

function rocblas_srotm_batched(handle, n, x, incx, y, incy, param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotm_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            param::Ptr{Ptr{Cfloat}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotm_batched(handle, n, x, incx, y, incy, param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotm_batched(handle::rocblas_handle, n::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            param::Ptr{Ptr{Cdouble}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y,
                                       param, stride_param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotm_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{Cfloat}, incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    param::Ptr{Cfloat},
                                                    stride_param::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y,
                                       param, stride_param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotm_strided_batched(handle::rocblas_handle, n::rocblas_int,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    param::Ptr{Cdouble},
                                                    stride_param::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotmg(handle, d1, d2, x1, y1, param)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotmg(handle::rocblas_handle, d1::Ptr{Cfloat},
                                     d2::Ptr{Cfloat}, x1::Ptr{Cfloat}, y1::Ptr{Cfloat},
                                     param::Ptr{Cfloat})::rocblas_status
end

function rocblas_drotmg(handle, d1, d2, x1, y1, param)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotmg(handle::rocblas_handle, d1::Ptr{Cdouble},
                                     d2::Ptr{Cdouble}, x1::Ptr{Cdouble}, y1::Ptr{Cdouble},
                                     param::Ptr{Cdouble})::rocblas_status
end

function rocblas_srotmg_batched(handle, d1, d2, x1, y1, param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotmg_batched(handle::rocblas_handle, d1::Ptr{Ptr{Cfloat}},
                                             d2::Ptr{Ptr{Cfloat}}, x1::Ptr{Ptr{Cfloat}},
                                             y1::Ptr{Ptr{Cfloat}}, param::Ptr{Ptr{Cfloat}},
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotmg_batched(handle, d1, d2, x1, y1, param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotmg_batched(handle::rocblas_handle, d1::Ptr{Ptr{Cdouble}},
                                             d2::Ptr{Ptr{Cdouble}}, x1::Ptr{Ptr{Cdouble}},
                                             y1::Ptr{Ptr{Cdouble}},
                                             param::Ptr{Ptr{Cdouble}},
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_srotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1,
                                        y1, stride_y1, param, stride_param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_srotmg_strided_batched(handle::rocblas_handle,
                                                     d1::Ptr{Cfloat},
                                                     stride_d1::rocblas_stride,
                                                     d2::Ptr{Cfloat},
                                                     stride_d2::rocblas_stride,
                                                     x1::Ptr{Cfloat},
                                                     stride_x1::rocblas_stride,
                                                     y1::Ptr{Cfloat},
                                                     stride_y1::rocblas_stride,
                                                     param::Ptr{Cfloat},
                                                     stride_param::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_drotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1,
                                        y1, stride_y1, param, stride_param, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_drotmg_strided_batched(handle::rocblas_handle,
                                                     d1::Ptr{Cdouble},
                                                     stride_d1::rocblas_stride,
                                                     d2::Ptr{Cdouble},
                                                     stride_d2::rocblas_stride,
                                                     x1::Ptr{Cdouble},
                                                     stride_x1::rocblas_stride,
                                                     y1::Ptr{Cdouble},
                                                     stride_y1::rocblas_stride,
                                                     param::Ptr{Cdouble},
                                                     stride_param::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgbmv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, kl::rocblas_int,
                                    ku::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                    lda::rocblas_int, x::Ptr{Cfloat}, incx::rocblas_int,
                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgbmv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, kl::rocblas_int,
                                    ku::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                    lda::rocblas_int, x::Ptr{Cdouble}, incx::rocblas_int,
                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_cgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgbmv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, kl::rocblas_int,
                                    ku::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgbmv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, kl::rocblas_int,
                                    ku::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_sgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y,
                               incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgbmv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, kl::rocblas_int,
                                            ku::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            beta::Ptr{Cfloat}, y::Ptr{Ptr{Cfloat}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y,
                               incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgbmv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, kl::rocblas_int,
                                            ku::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            beta::Ptr{Cdouble}, y::Ptr{Ptr{Cdouble}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y,
                               incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgbmv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, kl::rocblas_int,
                                            ku::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y,
                               incy, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgbmv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, kl::rocblas_int,
                                            ku::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A,
                                       x, incx, stride_x, beta, y, incy, stride_y,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgbmv_strided_batched(handle::rocblas_handle,
                                                    trans::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    kl::rocblas_int, ku::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A,
                                       x, incx, stride_x, beta, y, incy, stride_y,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgbmv_strided_batched(handle::rocblas_handle,
                                                    trans::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    kl::rocblas_int, ku::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A,
                                       x, incx, stride_x, beta, y, incy, stride_y,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgbmv_strided_batched(handle::rocblas_handle,
                                                    trans::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    kl::rocblas_int, ku::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A,
                                       x, incx, stride_x, beta, y, incy, stride_y,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgbmv_strided_batched(handle::rocblas_handle,
                                                    trans::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    kl::rocblas_int, ku::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, alpha::Ptr{Cfloat},
                                    A::Ptr{Cfloat}, lda::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int, alpha::Ptr{Cdouble},
                                    A::Ptr{Cdouble}, lda::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_cgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemv(handle::rocblas_handle, trans::rocblas_operation,
                                    m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_sgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            beta::Ptr{Cfloat}, y::Ptr{Ptr{Cfloat}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            beta::Ptr{Cdouble}, y::Ptr{Ptr{Cdouble}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemv_batched(handle::rocblas_handle,
                                            trans::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_hshgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                                 batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hshgemv_batched(handle::rocblas_handle,
                                              trans::rocblas_operation, m::rocblas_int,
                                              n::rocblas_int, alpha::Ptr{Cfloat},
                                              A::Ptr{Ptr{rocblas_half}}, lda::rocblas_int,
                                              x::Ptr{Ptr{rocblas_half}}, incx::rocblas_int,
                                              beta::Ptr{Cfloat}, y::Ptr{Ptr{rocblas_half}},
                                              incy::rocblas_int,
                                              batch_count::rocblas_int)::rocblas_status
end

function rocblas_hssgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                                 batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hssgemv_batched(handle::rocblas_handle,
                                              trans::rocblas_operation, m::rocblas_int,
                                              n::rocblas_int, alpha::Ptr{Cfloat},
                                              A::Ptr{Ptr{rocblas_half}}, lda::rocblas_int,
                                              x::Ptr{Ptr{rocblas_half}}, incx::rocblas_int,
                                              beta::Ptr{Cfloat}, y::Ptr{Ptr{Cfloat}},
                                              incy::rocblas_int,
                                              batch_count::rocblas_int)::rocblas_status
end

function rocblas_tstgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                                 batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_tstgemv_batched(handle::rocblas_handle,
                                              trans::rocblas_operation, m::rocblas_int,
                                              n::rocblas_int, alpha::Ptr{Cfloat},
                                              A::Ptr{Ptr{rocblas_bfloat16}},
                                              lda::rocblas_int,
                                              x::Ptr{Ptr{rocblas_bfloat16}},
                                              incx::rocblas_int, beta::Ptr{Cfloat},
                                              y::Ptr{Ptr{rocblas_bfloat16}},
                                              incy::rocblas_int,
                                              batch_count::rocblas_int)::rocblas_status
end

function rocblas_tssgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy,
                                 batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_tssgemv_batched(handle::rocblas_handle,
                                              trans::rocblas_operation, m::rocblas_int,
                                              n::rocblas_int, alpha::Ptr{Cfloat},
                                              A::Ptr{Ptr{rocblas_bfloat16}},
                                              lda::rocblas_int,
                                              x::Ptr{Ptr{rocblas_bfloat16}},
                                              incx::rocblas_int, beta::Ptr{Cfloat},
                                              y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                              batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                       incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemv_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                       incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemv_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                       incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemv_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                       incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemv_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_hshgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                         incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hshgemv_strided_batched(handle::rocblas_handle,
                                                      transA::rocblas_operation,
                                                      m::rocblas_int, n::rocblas_int,
                                                      alpha::Ptr{Cfloat},
                                                      A::Ptr{rocblas_half},
                                                      lda::rocblas_int,
                                                      strideA::rocblas_stride,
                                                      x::Ptr{rocblas_half},
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      beta::Ptr{Cfloat},
                                                      y::Ptr{rocblas_half},
                                                      incy::rocblas_int,
                                                      stridey::rocblas_stride,
                                                      batch_count::rocblas_int)::rocblas_status
end

function rocblas_hssgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                         incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hssgemv_strided_batched(handle::rocblas_handle,
                                                      transA::rocblas_operation,
                                                      m::rocblas_int, n::rocblas_int,
                                                      alpha::Ptr{Cfloat},
                                                      A::Ptr{rocblas_half},
                                                      lda::rocblas_int,
                                                      strideA::rocblas_stride,
                                                      x::Ptr{rocblas_half},
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                      incy::rocblas_int,
                                                      stridey::rocblas_stride,
                                                      batch_count::rocblas_int)::rocblas_status
end

function rocblas_tstgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                         incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_tstgemv_strided_batched(handle::rocblas_handle,
                                                      transA::rocblas_operation,
                                                      m::rocblas_int, n::rocblas_int,
                                                      alpha::Ptr{Cfloat},
                                                      A::Ptr{rocblas_bfloat16},
                                                      lda::rocblas_int,
                                                      strideA::rocblas_stride,
                                                      x::Ptr{rocblas_bfloat16},
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      beta::Ptr{Cfloat},
                                                      y::Ptr{rocblas_bfloat16},
                                                      incy::rocblas_int,
                                                      stridey::rocblas_stride,
                                                      batch_count::rocblas_int)::rocblas_status
end

function rocblas_tssgemv_strided_batched(handle, transA, m, n, alpha, A, lda, strideA, x,
                                         incx, stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_tssgemv_strided_batched(handle::rocblas_handle,
                                                      transA::rocblas_operation,
                                                      m::rocblas_int, n::rocblas_int,
                                                      alpha::Ptr{Cfloat},
                                                      A::Ptr{rocblas_bfloat16},
                                                      lda::rocblas_int,
                                                      strideA::rocblas_stride,
                                                      x::Ptr{rocblas_bfloat16},
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                      incy::rocblas_int,
                                                      stridey::rocblas_stride,
                                                      batch_count::rocblas_int)::rocblas_status
end

function rocblas_chbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, k::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, k::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_chbmv_batched(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhbmv_batched(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_chbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    k::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    k::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_chemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_chemv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhemv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_chemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cfloat},
                                   x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                   A::Ptr{rocblas_float_complex},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_zher(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cdouble},
                                   x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                   A::Ptr{rocblas_double_complex},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_cher_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cfloat},
                                           x::Ptr{Ptr{rocblas_float_complex}},
                                           incx::rocblas_int,
                                           A::Ptr{Ptr{rocblas_float_complex}},
                                           lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cdouble},
                                           x::Ptr{Ptr{rocblas_double_complex}},
                                           incx::rocblas_int,
                                           A::Ptr{Ptr{rocblas_double_complex}},
                                           lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, A, lda,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cfloat},
                                                   x::Ptr{rocblas_float_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   A::Ptr{rocblas_float_complex},
                                                   lda::rocblas_int,
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, A, lda,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cdouble},
                                                   x::Ptr{rocblas_double_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   A::Ptr{rocblas_double_complex},
                                                   lda::rocblas_int,
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_zher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_cher2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, A, lda, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, A, lda, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    AP::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    AP::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_chpmv_batched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            AP::Ptr{Ptr{rocblas_float_complex}},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpmv_batched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            AP::Ptr{Ptr{rocblas_double_complex}},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpmv_strided_batched(handle, uplo, n, alpha, AP, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    AP::Ptr{rocblas_float_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpmv_strided_batched(handle, uplo, n, alpha, AP, stride_A, x, incx,
                                       stride_x, beta, y, incy, stride_y, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    AP::Ptr{rocblas_double_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cfloat},
                                   x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                   AP::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zhpr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cdouble},
                                   x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                   AP::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_chpr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cfloat},
                                           x::Ptr{Ptr{rocblas_float_complex}},
                                           incx::rocblas_int,
                                           AP::Ptr{Ptr{rocblas_float_complex}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cdouble},
                                           x::Ptr{Ptr{rocblas_double_complex}},
                                           incx::rocblas_int,
                                           AP::Ptr{Ptr{rocblas_double_complex}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cfloat},
                                                   x::Ptr{rocblas_float_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{rocblas_float_complex},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cdouble},
                                                   x::Ptr{rocblas_double_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{rocblas_double_complex},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpr2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    AP::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zhpr2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    AP::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_chpr2_batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            AP::Ptr{Ptr{rocblas_float_complex}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpr2_batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            AP::Ptr{Ptr{rocblas_double_complex}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_chpr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, AP, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chpr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    AP::Ptr{rocblas_float_complex},
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhpr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, AP, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhpr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    AP::Ptr{rocblas_double_complex},
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_strmv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cfloat}, lda::rocblas_int,
                                    x::Ptr{Cfloat}, incx::rocblas_int)::rocblas_status
end

function rocblas_dtrmv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cdouble}, lda::rocblas_int,
                                    x::Ptr{Cdouble}, incx::rocblas_int)::rocblas_status
end

function rocblas_ctrmv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int, x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztrmv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int, x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_strmv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrmv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrmv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrmv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_strmv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrmv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrmv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrmv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_stpmv(handle, uplo, transA, diag, m, A, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cfloat}, x::Ptr{Cfloat},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_dtpmv(handle, uplo, transA, diag, m, A, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cdouble}, x::Ptr{Cdouble},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ctpmv(handle, uplo, transA, diag, m, A, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztpmv(handle, uplo, transA, diag, m, A, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_stpmv_batched(handle, uplo, transA, diag, m, A, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cfloat}}, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtpmv_batched(handle, uplo, transA, diag, m, A, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cdouble}}, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctpmv_batched(handle, uplo, transA, diag, m, A, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztpmv_batched(handle, uplo, transA, diag, m, A, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_stpmv_strided_batched(handle, uplo, transA, diag, m, A, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cfloat},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtpmv_strided_batched(handle, uplo, transA, diag, m, A, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cdouble},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctpmv_strided_batched(handle, uplo, transA, diag, m, A, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztpmv_strided_batched(handle, uplo, transA, diag, m, A, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_stbmv(handle, uplo, trans, diag, m, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    trans::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, k::rocblas_int, A::Ptr{Cfloat},
                                    lda::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_dtbmv(handle, uplo, trans, diag, m, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    trans::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, k::rocblas_int, A::Ptr{Cdouble},
                                    lda::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ctbmv(handle, uplo, trans, diag, m, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    trans::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, k::rocblas_int,
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztbmv(handle, uplo, trans, diag, m, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    trans::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, k::rocblas_int,
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_stbmv_batched(handle, uplo, trans, diag, m, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            trans::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            k::rocblas_int, A::Ptr{Ptr{Cfloat}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtbmv_batched(handle, uplo, trans, diag, m, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            trans::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            k::rocblas_int, A::Ptr{Ptr{Cdouble}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctbmv_batched(handle, uplo, trans, diag, m, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            trans::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            k::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztbmv_batched(handle, uplo, trans, diag, m, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            trans::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            k::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_stbmv_strided_batched(handle, uplo, trans, diag, m, k, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    trans::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    k::rocblas_int, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtbmv_strided_batched(handle, uplo, trans, diag, m, k, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    trans::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    k::rocblas_int, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctbmv_strided_batched(handle, uplo, trans, diag, m, k, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    trans::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    k::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztbmv_strided_batched(handle, uplo, trans, diag, m, k, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    trans::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    k::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_stbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, k::rocblas_int, A::Ptr{Cfloat},
                                    lda::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_dtbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, k::rocblas_int, A::Ptr{Cdouble},
                                    lda::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ctbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, k::rocblas_int,
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, k::rocblas_int,
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_stbsv_batched(handle, uplo, transA, diag, n, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            k::rocblas_int, A::Ptr{Ptr{Cfloat}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtbsv_batched(handle, uplo, transA, diag, n, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            k::rocblas_int, A::Ptr{Ptr{Cdouble}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctbsv_batched(handle, uplo, transA, diag, n, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            k::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztbsv_batched(handle, uplo, transA, diag, n, k, A, lda, x, incx,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            k::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_stbsv_strided_batched(handle, uplo, transA, diag, n, k, A, lda, stride_A,
                                       x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stbsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    k::rocblas_int, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtbsv_strided_batched(handle, uplo, transA, diag, n, k, A, lda, stride_A,
                                       x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtbsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    k::rocblas_int, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctbsv_strided_batched(handle, uplo, transA, diag, n, k, A, lda, stride_A,
                                       x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctbsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    k::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztbsv_strided_batched(handle, uplo, transA, diag, n, k, A, lda, stride_A,
                                       x, incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztbsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    k::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_strsv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cfloat}, lda::rocblas_int,
                                    x::Ptr{Cfloat}, incx::rocblas_int)::rocblas_status
end

function rocblas_dtrsv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{Cdouble}, lda::rocblas_int,
                                    x::Ptr{Cdouble}, incx::rocblas_int)::rocblas_status
end

function rocblas_ctrsv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int, x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztrsv(handle, uplo, transA, diag, m, A, lda, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    m::rocblas_int, A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int, x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_strsv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrsv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrsv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrsv_batched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_strsv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrsv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrsv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrsv_strided_batched(handle, uplo, transA, diag, m, A, lda, stride_A, x,
                                       incx, stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_stpsv(handle, uplo, transA, diag, n, AP, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, AP::Ptr{Cfloat}, x::Ptr{Cfloat},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_dtpsv(handle, uplo, transA, diag, n, AP, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, AP::Ptr{Cdouble}, x::Ptr{Cdouble},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ctpsv(handle, uplo, transA, diag, n, AP, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, AP::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_ztpsv(handle, uplo, transA, diag, n, AP, x, incx)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpsv(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, diag::rocblas_diagonal,
                                    n::rocblas_int, AP::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex},
                                    incx::rocblas_int)::rocblas_status
end

function rocblas_stpsv_batched(handle, uplo, transA, diag, n, AP, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            AP::Ptr{Ptr{Cfloat}}, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtpsv_batched(handle, uplo, transA, diag, n, AP, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            AP::Ptr{Ptr{Cdouble}}, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctpsv_batched(handle, uplo, transA, diag, n, AP, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            AP::Ptr{Ptr{rocblas_float_complex}},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztpsv_batched(handle, uplo, transA, diag, n, AP, x, incx, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpsv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation,
                                            diag::rocblas_diagonal, n::rocblas_int,
                                            AP::Ptr{Ptr{rocblas_double_complex}},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_stpsv_strided_batched(handle, uplo, transA, diag, n, AP, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stpsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    AP::Ptr{Cfloat},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtpsv_strided_batched(handle, uplo, transA, diag, n, AP, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtpsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    AP::Ptr{Cdouble},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctpsv_strided_batched(handle, uplo, transA, diag, n, AP, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctpsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    AP::Ptr{rocblas_float_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztpsv_strided_batched(handle, uplo, transA, diag, n, AP, stride_A, x, incx,
                                       stride_x, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztpsv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, n::rocblas_int,
                                                    AP::Ptr{rocblas_double_complex},
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                    lda::rocblas_int, x::Ptr{Cfloat}, incx::rocblas_int,
                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                    lda::rocblas_int, x::Ptr{Cdouble}, incx::rocblas_int,
                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_csymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    y::Ptr{rocblas_float_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_zsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    y::Ptr{rocblas_double_complex},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_ssymv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            beta::Ptr{Cfloat}, y::Ptr{Ptr{Cfloat}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsymv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            beta::Ptr{Cdouble}, y::Ptr{Ptr{Cdouble}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_csymv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsymv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssymv_strided_batched(handle, uplo, n, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsymv_strided_batched(handle, uplo, n, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_csymv_strided_batched(handle, uplo, n, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsymv_strided_batched(handle, uplo, n, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspmv(handle, uplo, n, alpha, A, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                    x::Ptr{Cfloat}, incx::rocblas_int, beta::Ptr{Cfloat},
                                    y::Ptr{Cfloat}, incy::rocblas_int)::rocblas_status
end

function rocblas_dspmv(handle, uplo, n, alpha, A, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                    x::Ptr{Cdouble}, incx::rocblas_int, beta::Ptr{Cdouble},
                                    y::Ptr{Cdouble}, incy::rocblas_int)::rocblas_status
end

function rocblas_sspmv_batched(handle, uplo, n, alpha, A, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int, beta::Ptr{Cfloat},
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspmv_batched(handle, uplo, n, alpha, A, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int, beta::Ptr{Cdouble},
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspmv_strided_batched(handle, uplo, n, alpha, A, strideA, x, incx, stridex,
                                       beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    strideA::rocblas_stride, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspmv_strided_batched(handle, uplo, n, alpha, A, strideA, x, incx, stridex,
                                       beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    strideA::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, k::rocblas_int, alpha::Ptr{Cfloat},
                                    A::Ptr{Cfloat}, lda::rocblas_int, x::Ptr{Cfloat},
                                    incx::rocblas_int, beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsbmv(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, k::rocblas_int, alpha::Ptr{Cdouble},
                                    A::Ptr{Cdouble}, lda::rocblas_int, x::Ptr{Cdouble},
                                    incx::rocblas_int, beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                    incy::rocblas_int)::rocblas_status
end

function rocblas_dsbmv_batched(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{Cdouble}, A::Ptr{Ptr{Cdouble}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cdouble}},
                                            incx::rocblas_int, beta::Ptr{Cdouble},
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssbmv_batched(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssbmv_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{Cfloat}, A::Ptr{Ptr{Cfloat}},
                                            lda::rocblas_int, x::Ptr{Ptr{Cfloat}},
                                            incx::rocblas_int, beta::Ptr{Cfloat},
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    k::rocblas_int, alpha::Ptr{Cfloat},
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    strideA::rocblas_stride, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cfloat}, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, strideA, x, incx,
                                       stridex, beta, y, incy, stridey, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsbmv_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    k::rocblas_int, alpha::Ptr{Cdouble},
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    beta::Ptr{Cdouble}, y::Ptr{Cdouble},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sger(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sger(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                   alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::rocblas_int,
                                   y::Ptr{Cfloat}, incy::rocblas_int, A::Ptr{Cfloat},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_dger(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dger(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                   alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::rocblas_int,
                                   y::Ptr{Cdouble}, incy::rocblas_int, A::Ptr{Cdouble},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_cgeru(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeru(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_zgeru(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeru(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_cgerc(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgerc(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_zgerc(handle, m, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgerc(handle::rocblas_handle, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_sger_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sger_batched(handle::rocblas_handle, m::rocblas_int,
                                           n::rocblas_int, alpha::Ptr{Cfloat},
                                           x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                           A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_dger_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dger_batched(handle::rocblas_handle, m::rocblas_int,
                                           n::rocblas_int, alpha::Ptr{Cdouble},
                                           x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                           y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                           A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgeru_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeru_batched(handle::rocblas_handle, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgeru_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeru_batched(handle::rocblas_handle, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgerc_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgerc_batched(handle::rocblas_handle, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgerc_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgerc_batched(handle::rocblas_handle, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sger_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                      stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sger_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                   n::rocblas_int, alpha::Ptr{Cfloat},
                                                   x::Ptr{Cfloat}, incx::rocblas_int,
                                                   stridex::rocblas_stride, y::Ptr{Cfloat},
                                                   incy::rocblas_int,
                                                   stridey::rocblas_stride, A::Ptr{Cfloat},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_dger_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                      stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dger_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                   n::rocblas_int, alpha::Ptr{Cdouble},
                                                   x::Ptr{Cdouble}, incx::rocblas_int,
                                                   stridex::rocblas_stride, y::Ptr{Cdouble},
                                                   incy::rocblas_int,
                                                   stridey::rocblas_stride, A::Ptr{Cdouble},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgeru_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeru_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgeru_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeru_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgerc_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgerc_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgerc_strided_batched(handle, m, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgerc_strided_batched(handle::rocblas_handle, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                   incx::rocblas_int, AP::Ptr{Cfloat})::rocblas_status
end

function rocblas_dspr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                   incx::rocblas_int, AP::Ptr{Cdouble})::rocblas_status
end

function rocblas_cspr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cspr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                   x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                   AP::Ptr{rocblas_float_complex})::rocblas_status
end

function rocblas_zspr(handle, uplo, n, alpha, x, incx, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zspr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                   x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                   AP::Ptr{rocblas_double_complex})::rocblas_status
end

function rocblas_sspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cfloat},
                                           x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                           AP::Ptr{Ptr{Cfloat}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cdouble},
                                           x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                           AP::Ptr{Ptr{Cdouble}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_cspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cspr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int,
                                           alpha::Ptr{rocblas_float_complex},
                                           x::Ptr{Ptr{rocblas_float_complex}},
                                           incx::rocblas_int,
                                           AP::Ptr{Ptr{rocblas_float_complex}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_zspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zspr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int,
                                           alpha::Ptr{rocblas_double_complex},
                                           x::Ptr{Ptr{rocblas_double_complex}},
                                           incx::rocblas_int,
                                           AP::Ptr{Ptr{rocblas_double_complex}},
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{Cfloat},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{Cdouble},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_cspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cspr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{rocblas_float_complex},
                                                   x::Ptr{rocblas_float_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{rocblas_float_complex},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_zspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, AP,
                                      stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zspr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{rocblas_double_complex},
                                                   x::Ptr{rocblas_double_complex},
                                                   incx::rocblas_int,
                                                   stride_x::rocblas_stride,
                                                   AP::Ptr{rocblas_double_complex},
                                                   stride_A::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspr2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                    incx::rocblas_int, y::Ptr{Cfloat}, incy::rocblas_int,
                                    AP::Ptr{Cfloat})::rocblas_status
end

function rocblas_dspr2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                    incx::rocblas_int, y::Ptr{Cdouble}, incy::rocblas_int,
                                    AP::Ptr{Cdouble})::rocblas_status
end

function rocblas_sspr2_batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            AP::Ptr{Ptr{Cfloat}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspr2_batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            AP::Ptr{Ptr{Cdouble}},
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sspr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, AP, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sspr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{Cfloat}, incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    AP::Ptr{Cfloat},
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dspr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, y, incy,
                                       stride_y, AP, stride_A, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dspr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stride_y::rocblas_stride,
                                                    AP::Ptr{Cdouble},
                                                    stride_A::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                   incx::rocblas_int, A::Ptr{Cfloat},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_dsyr(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                   incx::rocblas_int, A::Ptr{Cdouble},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_csyr(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                   x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                   A::Ptr{rocblas_float_complex},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_zsyr(handle, uplo, n, alpha, x, incx, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr(handle::rocblas_handle, uplo::rocblas_fill,
                                   n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                   x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                   A::Ptr{rocblas_double_complex},
                                   lda::rocblas_int)::rocblas_status
end

function rocblas_ssyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cfloat},
                                           x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                           A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int, alpha::Ptr{Cdouble},
                                           x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                           A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int,
                                           alpha::Ptr{rocblas_float_complex},
                                           x::Ptr{Ptr{rocblas_float_complex}},
                                           incx::rocblas_int,
                                           A::Ptr{Ptr{rocblas_float_complex}},
                                           lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                           n::rocblas_int,
                                           alpha::Ptr{rocblas_double_complex},
                                           x::Ptr{Ptr{rocblas_double_complex}},
                                           incx::rocblas_int,
                                           A::Ptr{Ptr{rocblas_double_complex}},
                                           lda::rocblas_int,
                                           batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr_strided_batched(handle, uplo, n, alpha, x, incx, stridex, A, lda,
                                      strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                   incx::rocblas_int,
                                                   stridex::rocblas_stride, A::Ptr{Cfloat},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr_strided_batched(handle, uplo, n, alpha, x, incx, stridex, A, lda,
                                      strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                   incx::rocblas_int,
                                                   stridex::rocblas_stride, A::Ptr{Cdouble},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr_strided_batched(handle, uplo, n, alpha, x, incx, stridex, A, lda,
                                      strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{rocblas_float_complex},
                                                   x::Ptr{rocblas_float_complex},
                                                   incx::rocblas_int,
                                                   stridex::rocblas_stride,
                                                   A::Ptr{rocblas_float_complex},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr_strided_batched(handle, uplo, n, alpha, x, incx, stridex, A, lda,
                                      strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr_strided_batched(handle::rocblas_handle,
                                                   uplo::rocblas_fill, n::rocblas_int,
                                                   alpha::Ptr{rocblas_double_complex},
                                                   x::Ptr{rocblas_double_complex},
                                                   incx::rocblas_int,
                                                   stridex::rocblas_stride,
                                                   A::Ptr{rocblas_double_complex},
                                                   lda::rocblas_int,
                                                   strideA::rocblas_stride,
                                                   batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                    incx::rocblas_int, y::Ptr{Cfloat}, incy::rocblas_int,
                                    A::Ptr{Cfloat}, lda::rocblas_int)::rocblas_status
end

function rocblas_dsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                    incx::rocblas_int, y::Ptr{Cdouble}, incy::rocblas_int,
                                    A::Ptr{Cdouble}, lda::rocblas_int)::rocblas_status
end

function rocblas_csyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_float_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_float_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_zsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2(handle::rocblas_handle, uplo::rocblas_fill,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    y::Ptr{rocblas_double_complex}, incy::rocblas_int,
                                    A::Ptr{rocblas_double_complex},
                                    lda::rocblas_int)::rocblas_status
end

function rocblas_ssyr2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cfloat}}, incy::rocblas_int,
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            y::Ptr{Ptr{Cdouble}}, incy::rocblas_int,
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_float_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr2_batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            y::Ptr{Ptr{rocblas_double_complex}},
                                            incy::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr2_strided_batched(handle, uplo, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride, y::Ptr{Cfloat},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr2_strided_batched(handle, uplo, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{Cdouble}, incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr2_strided_batched(handle, uplo, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_float_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr2_strided_batched(handle, uplo, n, alpha, x, incx, stridex, y, incy,
                                       stridey, A, lda, strideA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stridex::rocblas_stride,
                                                    y::Ptr{rocblas_double_complex},
                                                    incy::rocblas_int,
                                                    stridey::rocblas_stride,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    strideA::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_chemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zhemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_chemm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhemm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_chemm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_chemm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zhemm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zhemm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{Cfloat},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    beta::Ptr{Cfloat}, C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{Cdouble},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    beta::Ptr{Cdouble}, C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_cherk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int, beta::Ptr{Cfloat},
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zherk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int, beta::Ptr{Cdouble},
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cherk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{Cfloat},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cfloat},
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zherk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{Cdouble},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cdouble},
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                     beta::Ptr{Cfloat}, C::Ptr{rocblas_float_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_zher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                     beta::Ptr{Cdouble}, C::Ptr{rocblas_double_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_cher2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_float_complex},
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_float_complex}},
                                             ldb::rocblas_int, beta::Ptr{Cfloat},
                                             C::Ptr{Ptr{rocblas_float_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_double_complex},
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_double_complex}},
                                             ldb::rocblas_int, beta::Ptr{Cdouble},
                                             C::Ptr{Ptr{rocblas_double_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_cher2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cher2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_float_complex},
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_float_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cfloat},
                                                     C::Ptr{rocblas_float_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zher2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zher2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_double_complex},
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_double_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cdouble},
                                                     C::Ptr{rocblas_double_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_cherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                     beta::Ptr{Cfloat}, C::Ptr{rocblas_float_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_zherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                     beta::Ptr{Cdouble}, C::Ptr{rocblas_double_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_cherkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_float_complex},
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_float_complex}},
                                             ldb::rocblas_int, beta::Ptr{Cfloat},
                                             C::Ptr{Ptr{rocblas_float_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zherkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_double_complex},
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_double_complex}},
                                             ldb::rocblas_int, beta::Ptr{Cdouble},
                                             C::Ptr{Ptr{rocblas_double_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_cherkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cherkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_float_complex},
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_float_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cfloat},
                                                     C::Ptr{rocblas_float_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zherkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zherkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_double_complex},
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_double_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cdouble},
                                                     C::Ptr{rocblas_double_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::rocblas_int,
                                    B::Ptr{Cfloat}, ldb::rocblas_int, beta::Ptr{Cfloat},
                                    C::Ptr{Cfloat}, ldc::rocblas_int)::rocblas_status
end

function rocblas_dsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::rocblas_int,
                                    B::Ptr{Cdouble}, ldb::rocblas_int, beta::Ptr{Cdouble},
                                    C::Ptr{Cdouble}, ldc::rocblas_int)::rocblas_status
end

function rocblas_csymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_ssymm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                            beta::Ptr{Cfloat}, C::Ptr{Ptr{Cfloat}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsymm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                            beta::Ptr{Cdouble}, C::Ptr{Ptr{Cdouble}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_csymm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsymm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssymm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssymm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{Cfloat}, ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsymm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsymm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{Cdouble}, ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_csymm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csymm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsymm_strided_batched(handle, side, uplo, m, n, alpha, A, lda, stride_A, B,
                                       ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsymm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                    lda::rocblas_int, beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_dsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                    lda::rocblas_int, beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_csyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrk(handle::rocblas_handle, uplo::rocblas_fill,
                                    transA::rocblas_operation, n::rocblas_int,
                                    k::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_ssyrk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            beta::Ptr{Cfloat}, C::Ptr{Ptr{Cfloat}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyrk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            beta::Ptr{Cdouble}, C::Ptr{Ptr{Cdouble}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyrk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyrk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc,
                               batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrk_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                            transA::rocblas_operation, n::rocblas_int,
                                            k::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyrk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyrk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyrk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyrk_strided_batched(handle, uplo, transA, n, k, alpha, A, lda, stride_A,
                                       beta, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrk_strided_batched(handle::rocblas_handle,
                                                    uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    n::rocblas_int, k::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                     lda::rocblas_int, B::Ptr{Cfloat}, ldb::rocblas_int,
                                     beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_dsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                     lda::rocblas_int, B::Ptr{Cdouble}, ldb::rocblas_int,
                                     beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_csyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_float_complex},
                                     C::Ptr{rocblas_float_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_zsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2k(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_double_complex},
                                     C::Ptr{rocblas_double_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_ssyr2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cfloat},
                                             A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                             beta::Ptr{Cfloat}, C::Ptr{Ptr{Cfloat}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cdouble},
                                             A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                             beta::Ptr{Cdouble}, C::Ptr{Ptr{Cdouble}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_float_complex},
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_float_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_float_complex},
                                             C::Ptr{Ptr{rocblas_float_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr2k_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2k_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_double_complex},
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_double_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_double_complex},
                                             C::Ptr{Ptr{rocblas_double_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyr2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyr2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{Cfloat}, ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyr2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyr2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{Cdouble}, ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyr2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyr2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_float_complex},
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_float_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{rocblas_float_complex},
                                                     C::Ptr{rocblas_float_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyr2k_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyr2k_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_double_complex},
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_double_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{rocblas_double_complex},
                                                     C::Ptr{rocblas_double_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                     lda::rocblas_int, B::Ptr{Cfloat}, ldb::rocblas_int,
                                     beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_dsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                     lda::rocblas_int, B::Ptr{Cdouble}, ldb::rocblas_int,
                                     beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_csyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_float_complex},
                                     C::Ptr{rocblas_float_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_zsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrkx(handle::rocblas_handle, uplo::rocblas_fill,
                                     trans::rocblas_operation, n::rocblas_int,
                                     k::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_double_complex},
                                     C::Ptr{rocblas_double_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_ssyrkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cfloat},
                                             A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                             beta::Ptr{Cfloat}, C::Ptr{Ptr{Cfloat}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyrkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cdouble},
                                             A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                             beta::Ptr{Cdouble}, C::Ptr{Ptr{Cdouble}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyrkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_float_complex},
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_float_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_float_complex},
                                             C::Ptr{Ptr{rocblas_float_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyrkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,
                                ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrkx_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             trans::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_double_complex},
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_double_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_double_complex},
                                             C::Ptr{Ptr{rocblas_double_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_ssyrkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ssyrkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{Cfloat}, ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_dsyrkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dsyrkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{Cdouble}, ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_csyrkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_csyrkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_float_complex},
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_float_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{rocblas_float_complex},
                                                     C::Ptr{rocblas_float_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zsyrkx_strided_batched(handle, uplo, trans, n, k, alpha, A, lda, stride_A,
                                        B, ldb, stride_B, beta, C, ldc, stride_C,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zsyrkx_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     trans::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_double_complex},
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_A::rocblas_stride,
                                                     B::Ptr{rocblas_double_complex},
                                                     ldb::rocblas_int,
                                                     stride_B::rocblas_stride,
                                                     beta::Ptr{rocblas_double_complex},
                                                     C::Ptr{rocblas_double_complex},
                                                     ldc::rocblas_int,
                                                     stride_C::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_strmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C,
                       ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::rocblas_int,
                                    B::Ptr{Cfloat}, ldb::rocblas_int, C::Ptr{Cfloat},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_dtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C,
                       ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::rocblas_int,
                                    B::Ptr{Cdouble}, ldb::rocblas_int, C::Ptr{Cdouble},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_ctrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C,
                       ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_ztrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C,
                       ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_strmm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                            C::Ptr{Ptr{Cfloat}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                            C::Ptr{Ptr{Cdouble}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_strmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_A, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int, alpha::Ptr{Cfloat},
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{Cfloat}, ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{Cfloat}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_A, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int, alpha::Ptr{Cdouble},
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{Cdouble}, ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{Cdouble}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_A, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_A, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_strtri(handle, uplo, diag, n, A, lda, invA, ldinvA)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strtri(handle::rocblas_handle, uplo::rocblas_fill,
                                     diag::rocblas_diagonal, n::rocblas_int, A::Ptr{Cfloat},
                                     lda::rocblas_int, invA::Ptr{Cfloat},
                                     ldinvA::rocblas_int)::rocblas_status
end

function rocblas_dtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrtri(handle::rocblas_handle, uplo::rocblas_fill,
                                     diag::rocblas_diagonal, n::rocblas_int,
                                     A::Ptr{Cdouble}, lda::rocblas_int, invA::Ptr{Cdouble},
                                     ldinvA::rocblas_int)::rocblas_status
end

function rocblas_ctrtri(handle, uplo, diag, n, A, lda, invA, ldinvA)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrtri(handle::rocblas_handle, uplo::rocblas_fill,
                                     diag::rocblas_diagonal, n::rocblas_int,
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     invA::Ptr{rocblas_float_complex},
                                     ldinvA::rocblas_int)::rocblas_status
end

function rocblas_ztrtri(handle, uplo, diag, n, A, lda, invA, ldinvA)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrtri(handle::rocblas_handle, uplo::rocblas_fill,
                                     diag::rocblas_diagonal, n::rocblas_int,
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     invA::Ptr{rocblas_double_complex},
                                     ldinvA::rocblas_int)::rocblas_status
end

function rocblas_strtri_batched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strtri_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             diag::rocblas_diagonal, n::rocblas_int,
                                             A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                             invA::Ptr{Ptr{Cfloat}}, ldinvA::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrtri_batched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrtri_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             diag::rocblas_diagonal, n::rocblas_int,
                                             A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                             invA::Ptr{Ptr{Cdouble}}, ldinvA::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrtri_batched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrtri_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             diag::rocblas_diagonal, n::rocblas_int,
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             invA::Ptr{Ptr{rocblas_float_complex}},
                                             ldinvA::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrtri_batched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrtri_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             diag::rocblas_diagonal, n::rocblas_int,
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             invA::Ptr{Ptr{rocblas_double_complex}},
                                             ldinvA::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_strtri_strided_batched(handle, uplo, diag, n, A, lda, stride_a, invA,
                                        ldinvA, stride_invA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strtri_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     diag::rocblas_diagonal, n::rocblas_int,
                                                     A::Ptr{Cfloat}, lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     invA::Ptr{Cfloat}, ldinvA::rocblas_int,
                                                     stride_invA::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrtri_strided_batched(handle, uplo, diag, n, A, lda, stride_a, invA,
                                        ldinvA, stride_invA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrtri_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     diag::rocblas_diagonal, n::rocblas_int,
                                                     A::Ptr{Cdouble}, lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     invA::Ptr{Cdouble},
                                                     ldinvA::rocblas_int,
                                                     stride_invA::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrtri_strided_batched(handle, uplo, diag, n, A, lda, stride_a, invA,
                                        ldinvA, stride_invA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrtri_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     diag::rocblas_diagonal, n::rocblas_int,
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     invA::Ptr{rocblas_float_complex},
                                                     ldinvA::rocblas_int,
                                                     stride_invA::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrtri_strided_batched(handle, uplo, diag, n, A, lda, stride_a, invA,
                                        ldinvA, stride_invA, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrtri_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     diag::rocblas_diagonal, n::rocblas_int,
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     invA::Ptr{rocblas_double_complex},
                                                     ldinvA::rocblas_int,
                                                     stride_invA::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::rocblas_int,
                                    B::Ptr{Cfloat}, ldb::rocblas_int)::rocblas_status
end

function rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::rocblas_int,
                                    B::Ptr{Cdouble}, ldb::rocblas_int)::rocblas_status
end

function rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_float_complex},
                                    ldb::rocblas_int)::rocblas_status
end

function rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsm(handle::rocblas_handle, side::rocblas_side,
                                    uplo::rocblas_fill, transA::rocblas_operation,
                                    diag::rocblas_diagonal, m::rocblas_int, n::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_double_complex},
                                    ldb::rocblas_int)::rocblas_status
end

function rocblas_strsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                               ldb, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsm_batched(handle::rocblas_handle, side::rocblas_side,
                                            uplo::rocblas_fill, transA::rocblas_operation,
                                            diag::rocblas_diagonal, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_strsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_a, B, ldb, stride_b, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_strsm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int, alpha::Ptr{Cfloat},
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{Cfloat}, ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dtrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_a, B, ldb, stride_b, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dtrsm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int, alpha::Ptr{Cdouble},
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{Cdouble}, ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ctrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_a, B, ldb, stride_b, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ctrsm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ztrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, A,
                                       lda, stride_a, B, ldb, stride_b, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ztrsm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, uplo::rocblas_fill,
                                                    transA::rocblas_operation,
                                                    diag::rocblas_diagonal, m::rocblas_int,
                                                    n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemm(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, k::rocblas_int, alpha::Ptr{Cfloat},
                                    A::Ptr{Cfloat}, lda::rocblas_int, B::Ptr{Cfloat},
                                    ldb::rocblas_int, beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemm(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, k::rocblas_int, alpha::Ptr{Cdouble},
                                    A::Ptr{Cdouble}, lda::rocblas_int, B::Ptr{Cdouble},
                                    ldb::rocblas_int, beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_hgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hgemm(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, k::rocblas_int,
                                    alpha::Ptr{rocblas_half}, A::Ptr{rocblas_half},
                                    lda::rocblas_int, B::Ptr{rocblas_half},
                                    ldb::rocblas_int, beta::Ptr{rocblas_half},
                                    C::Ptr{rocblas_half}, ldc::rocblas_int)::rocblas_status
end

function rocblas_cgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemm(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, k::rocblas_int,
                                    alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemm(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, k::rocblas_int,
                                    alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_sgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta,
                               C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemm_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{Cfloat}, A::Ptr{Ptr{Cfloat}},
                                            lda::rocblas_int, B::Ptr{Ptr{Cfloat}},
                                            ldb::rocblas_int, beta::Ptr{Cfloat},
                                            C::Ptr{Ptr{Cfloat}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta,
                               C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemm_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{Cdouble}, A::Ptr{Ptr{Cdouble}},
                                            lda::rocblas_int, B::Ptr{Ptr{Cdouble}},
                                            ldb::rocblas_int, beta::Ptr{Cdouble},
                                            C::Ptr{Ptr{Cdouble}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_hgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta,
                               C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hgemm_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{rocblas_half},
                                            A::Ptr{Ptr{rocblas_half}}, lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_half}}, ldb::rocblas_int,
                                            beta::Ptr{rocblas_half},
                                            C::Ptr{Ptr{rocblas_half}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta,
                               C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemm_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta,
                               C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemm_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, k::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda,
                                       stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemm_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    k::rocblas_int, alpha::Ptr{Cfloat},
                                                    A::Ptr{Cfloat}, lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{Cfloat}, ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                    ldc::rocblas_int,
                                                    stride_c::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda,
                                       stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemm_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    k::rocblas_int, alpha::Ptr{Cdouble},
                                                    A::Ptr{Cdouble}, lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{Cdouble}, ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                    ldc::rocblas_int,
                                                    stride_c::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_hgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda,
                                       stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hgemm_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    k::rocblas_int,
                                                    alpha::Ptr{rocblas_half},
                                                    A::Ptr{rocblas_half}, lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{rocblas_half}, ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    beta::Ptr{rocblas_half},
                                                    C::Ptr{rocblas_half}, ldc::rocblas_int,
                                                    stride_c::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_hgemm_kernel_name(handle, transA, transB, m, n, k, alpha, A, lda, stride_a,
                                   B, ldb, stride_b, beta, C, ldc, stride_c, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_hgemm_kernel_name(handle::rocblas_handle,
                                                transA::rocblas_operation,
                                                transB::rocblas_operation, m::rocblas_int,
                                                n::rocblas_int, k::rocblas_int,
                                                alpha::Ptr{rocblas_half},
                                                A::Ptr{rocblas_half}, lda::rocblas_int,
                                                stride_a::rocblas_stride,
                                                B::Ptr{rocblas_half}, ldb::rocblas_int,
                                                stride_b::rocblas_stride,
                                                beta::Ptr{rocblas_half},
                                                C::Ptr{rocblas_half}, ldc::rocblas_int,
                                                stride_c::rocblas_stride,
                                                batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemm_kernel_name(handle, transA, transB, m, n, k, alpha, A, lda, stride_a,
                                   B, ldb, stride_b, beta, C, ldc, stride_c, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemm_kernel_name(handle::rocblas_handle,
                                                transA::rocblas_operation,
                                                transB::rocblas_operation, m::rocblas_int,
                                                n::rocblas_int, k::rocblas_int,
                                                alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                lda::rocblas_int, stride_a::rocblas_stride,
                                                B::Ptr{Cfloat}, ldb::rocblas_int,
                                                stride_b::rocblas_stride, beta::Ptr{Cfloat},
                                                C::Ptr{Cfloat}, ldc::rocblas_int,
                                                stride_c::rocblas_stride,
                                                batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemm_kernel_name(handle, transA, transB, m, n, k, alpha, A, lda, stride_a,
                                   B, ldb, stride_b, beta, C, ldc, stride_c, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemm_kernel_name(handle::rocblas_handle,
                                                transA::rocblas_operation,
                                                transB::rocblas_operation, m::rocblas_int,
                                                n::rocblas_int, k::rocblas_int,
                                                alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                lda::rocblas_int, stride_a::rocblas_stride,
                                                B::Ptr{Cdouble}, ldb::rocblas_int,
                                                stride_b::rocblas_stride,
                                                beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                ldc::rocblas_int, stride_c::rocblas_stride,
                                                batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda,
                                       stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemm_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    k::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_c::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda,
                                       stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemm_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    k::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_a::rocblas_stride,
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_b::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_c::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sdgmm(handle, side, m, n, A, lda, x, incx, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdgmm(handle::rocblas_handle, side::rocblas_side,
                                    m::rocblas_int, n::rocblas_int, A::Ptr{Cfloat},
                                    lda::rocblas_int, x::Ptr{Cfloat}, incx::rocblas_int,
                                    C::Ptr{Cfloat}, ldc::rocblas_int)::rocblas_status
end

function rocblas_ddgmm(handle, side, m, n, A, lda, x, incx, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddgmm(handle::rocblas_handle, side::rocblas_side,
                                    m::rocblas_int, n::rocblas_int, A::Ptr{Cdouble},
                                    lda::rocblas_int, x::Ptr{Cdouble}, incx::rocblas_int,
                                    C::Ptr{Cdouble}, ldc::rocblas_int)::rocblas_status
end

function rocblas_cdgmm(handle, side, m, n, A, lda, x, incx, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdgmm(handle::rocblas_handle, side::rocblas_side,
                                    m::rocblas_int, n::rocblas_int,
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_float_complex}, incx::rocblas_int,
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zdgmm(handle, side, m, n, A, lda, x, incx, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdgmm(handle::rocblas_handle, side::rocblas_side,
                                    m::rocblas_int, n::rocblas_int,
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    x::Ptr{rocblas_double_complex}, incx::rocblas_int,
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_sdgmm_batched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdgmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            m::rocblas_int, n::rocblas_int,
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cfloat}}, incx::rocblas_int,
                                            C::Ptr{Ptr{Cfloat}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_ddgmm_batched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddgmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            m::rocblas_int, n::rocblas_int,
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            x::Ptr{Ptr{Cdouble}}, incx::rocblas_int,
                                            C::Ptr{Ptr{Cdouble}}, ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cdgmm_batched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdgmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            m::rocblas_int, n::rocblas_int,
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_float_complex}},
                                            incx::rocblas_int,
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdgmm_batched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdgmm_batched(handle::rocblas_handle, side::rocblas_side,
                                            m::rocblas_int, n::rocblas_int,
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            x::Ptr{Ptr{rocblas_double_complex}},
                                            incx::rocblas_int,
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sdgmm_strided_batched(handle, side, m, n, A, lda, stride_A, x, incx,
                                       stride_x, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sdgmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, m::rocblas_int,
                                                    n::rocblas_int, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cfloat}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    C::Ptr{Cfloat}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_ddgmm_strided_batched(handle, side, m, n, A, lda, stride_A, x, incx,
                                       stride_x, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_ddgmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, m::rocblas_int,
                                                    n::rocblas_int, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{Cdouble}, incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    C::Ptr{Cdouble}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cdgmm_strided_batched(handle, side, m, n, A, lda, stride_A, x, incx,
                                       stride_x, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cdgmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, m::rocblas_int,
                                                    n::rocblas_int,
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_float_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zdgmm_strided_batched(handle, side, m, n, A, lda, stride_A, x, incx,
                                       stride_x, C, ldc, stride_C, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zdgmm_strided_batched(handle::rocblas_handle,
                                                    side::rocblas_side, m::rocblas_int,
                                                    n::rocblas_int,
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    x::Ptr{rocblas_double_complex},
                                                    incx::rocblas_int,
                                                    stride_x::rocblas_stride,
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgeam(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                    lda::rocblas_int, beta::Ptr{Cfloat}, B::Ptr{Cfloat},
                                    ldb::rocblas_int, C::Ptr{Cfloat},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_dgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgeam(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                    lda::rocblas_int, beta::Ptr{Cdouble}, B::Ptr{Cdouble},
                                    ldb::rocblas_int, C::Ptr{Cdouble},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_cgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeam(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, alpha::Ptr{rocblas_float_complex},
                                    A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                    beta::Ptr{rocblas_float_complex},
                                    B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                    C::Ptr{rocblas_float_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_zgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeam(handle::rocblas_handle, transA::rocblas_operation,
                                    transB::rocblas_operation, m::rocblas_int,
                                    n::rocblas_int, alpha::Ptr{rocblas_double_complex},
                                    A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                    beta::Ptr{rocblas_double_complex},
                                    B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                    C::Ptr{rocblas_double_complex},
                                    ldc::rocblas_int)::rocblas_status
end

function rocblas_sgeam_batched(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgeam_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cfloat},
                                            A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                            beta::Ptr{Cfloat}, B::Ptr{Ptr{Cfloat}},
                                            ldb::rocblas_int, C::Ptr{Ptr{Cfloat}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgeam_batched(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgeam_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int, alpha::Ptr{Cdouble},
                                            A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                            beta::Ptr{Cdouble}, B::Ptr{Ptr{Cdouble}},
                                            ldb::rocblas_int, C::Ptr{Ptr{Cdouble}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgeam_batched(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeam_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_float_complex},
                                            A::Ptr{Ptr{rocblas_float_complex}},
                                            lda::rocblas_int,
                                            beta::Ptr{rocblas_float_complex},
                                            B::Ptr{Ptr{rocblas_float_complex}},
                                            ldb::rocblas_int,
                                            C::Ptr{Ptr{rocblas_float_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgeam_batched(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C,
                               ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeam_batched(handle::rocblas_handle,
                                            transA::rocblas_operation,
                                            transB::rocblas_operation, m::rocblas_int,
                                            n::rocblas_int,
                                            alpha::Ptr{rocblas_double_complex},
                                            A::Ptr{Ptr{rocblas_double_complex}},
                                            lda::rocblas_int,
                                            beta::Ptr{rocblas_double_complex},
                                            B::Ptr{Ptr{rocblas_double_complex}},
                                            ldb::rocblas_int,
                                            C::Ptr{Ptr{rocblas_double_complex}},
                                            ldc::rocblas_int,
                                            batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgeam_strided_batched(handle, transA, transB, m, n, alpha, A, lda,
                                       stride_A, beta, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgeam_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cfloat}, B::Ptr{Cfloat},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{Cfloat}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgeam_strided_batched(handle, transA, transB, m, n, alpha, A, lda,
                                       stride_A, beta, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgeam_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{Cdouble}, B::Ptr{Cdouble},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{Cdouble}, ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgeam_strided_batched(handle, transA, transB, m, n, alpha, A, lda,
                                       stride_A, beta, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgeam_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_float_complex},
                                                    A::Ptr{rocblas_float_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{rocblas_float_complex},
                                                    B::Ptr{rocblas_float_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{rocblas_float_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgeam_strided_batched(handle, transA, transB, m, n, alpha, A, lda,
                                       stride_A, beta, B, ldb, stride_B, C, ldc, stride_C,
                                       batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgeam_strided_batched(handle::rocblas_handle,
                                                    transA::rocblas_operation,
                                                    transB::rocblas_operation,
                                                    m::rocblas_int, n::rocblas_int,
                                                    alpha::Ptr{rocblas_double_complex},
                                                    A::Ptr{rocblas_double_complex},
                                                    lda::rocblas_int,
                                                    stride_A::rocblas_stride,
                                                    beta::Ptr{rocblas_double_complex},
                                                    B::Ptr{rocblas_double_complex},
                                                    ldb::rocblas_int,
                                                    stride_B::rocblas_stride,
                                                    C::Ptr{rocblas_double_complex},
                                                    ldc::rocblas_int,
                                                    stride_C::rocblas_stride,
                                                    batch_count::rocblas_int)::rocblas_status
end

function rocblas_gemm_batched_ex(handle, transA, transB, m, n, k, alpha, a, a_type, lda, b,
                                 b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd,
                                 batch_count, compute_type, algo, solution_index, flags)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_gemm_batched_ex(handle::rocblas_handle,
                                              transA::rocblas_operation,
                                              transB::rocblas_operation, m::rocblas_int,
                                              n::rocblas_int, k::rocblas_int,
                                              alpha::Ptr{Cvoid}, a::Ptr{Cvoid},
                                              a_type::rocblas_datatype, lda::rocblas_int,
                                              b::Ptr{Cvoid}, b_type::rocblas_datatype,
                                              ldb::rocblas_int, beta::Ptr{Cvoid},
                                              c::Ptr{Cvoid}, c_type::rocblas_datatype,
                                              ldc::rocblas_int, d::Ptr{Cvoid},
                                              d_type::rocblas_datatype, ldd::rocblas_int,
                                              batch_count::rocblas_int,
                                              compute_type::rocblas_datatype,
                                              algo::rocblas_gemm_algo,
                                              solution_index::Int32,
                                              flags::UInt32)::rocblas_status
end

function rocblas_sgemmt(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemmt(handle::rocblas_handle, uplo::rocblas_fill,
                                     transA::rocblas_operation, transB::rocblas_operation,
                                     n::rocblas_int, k::rocblas_int, alpha::Ptr{Cfloat},
                                     A::Ptr{Cfloat}, lda::rocblas_int, B::Ptr{Cfloat},
                                     ldb::rocblas_int, beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_dgemmt(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemmt(handle::rocblas_handle, uplo::rocblas_fill,
                                     transA::rocblas_operation, transB::rocblas_operation,
                                     n::rocblas_int, k::rocblas_int, alpha::Ptr{Cdouble},
                                     A::Ptr{Cdouble}, lda::rocblas_int, B::Ptr{Cdouble},
                                     ldb::rocblas_int, beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_cgemmt(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemmt(handle::rocblas_handle, uplo::rocblas_fill,
                                     transA::rocblas_operation, transB::rocblas_operation,
                                     n::rocblas_int, k::rocblas_int,
                                     alpha::Ptr{rocblas_float_complex},
                                     A::Ptr{rocblas_float_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_float_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_float_complex},
                                     C::Ptr{rocblas_float_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_zgemmt(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemmt(handle::rocblas_handle, uplo::rocblas_fill,
                                     transA::rocblas_operation, transB::rocblas_operation,
                                     n::rocblas_int, k::rocblas_int,
                                     alpha::Ptr{rocblas_double_complex},
                                     A::Ptr{rocblas_double_complex}, lda::rocblas_int,
                                     B::Ptr{rocblas_double_complex}, ldb::rocblas_int,
                                     beta::Ptr{rocblas_double_complex},
                                     C::Ptr{rocblas_double_complex},
                                     ldc::rocblas_int)::rocblas_status
end

function rocblas_sgemmt_batched(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb,
                                beta, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemmt_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             transA::rocblas_operation,
                                             transB::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cfloat},
                                             A::Ptr{Ptr{Cfloat}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cfloat}}, ldb::rocblas_int,
                                             beta::Ptr{Cfloat}, C::Ptr{Ptr{Cfloat}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemmt_batched(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb,
                                beta, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemmt_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             transA::rocblas_operation,
                                             transB::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int, alpha::Ptr{Cdouble},
                                             A::Ptr{Ptr{Cdouble}}, lda::rocblas_int,
                                             B::Ptr{Ptr{Cdouble}}, ldb::rocblas_int,
                                             beta::Ptr{Cdouble}, C::Ptr{Ptr{Cdouble}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemmt_batched(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb,
                                beta, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemmt_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             transA::rocblas_operation,
                                             transB::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_float_complex},
                                             A::Ptr{Ptr{rocblas_float_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_float_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_float_complex},
                                             C::Ptr{Ptr{rocblas_float_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemmt_batched(handle, uplo, transA, transB, n, k, alpha, A, lda, B, ldb,
                                beta, C, ldc, batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemmt_batched(handle::rocblas_handle, uplo::rocblas_fill,
                                             transA::rocblas_operation,
                                             transB::rocblas_operation, n::rocblas_int,
                                             k::rocblas_int,
                                             alpha::Ptr{rocblas_double_complex},
                                             A::Ptr{Ptr{rocblas_double_complex}},
                                             lda::rocblas_int,
                                             B::Ptr{Ptr{rocblas_double_complex}},
                                             ldb::rocblas_int,
                                             beta::Ptr{rocblas_double_complex},
                                             C::Ptr{Ptr{rocblas_double_complex}},
                                             ldc::rocblas_int,
                                             batch_count::rocblas_int)::rocblas_status
end

function rocblas_sgemmt_strided_batched(handle, uplo, transA, transB, n, k, alpha, A, lda,
                                        stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_sgemmt_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     transA::rocblas_operation,
                                                     transB::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cfloat}, A::Ptr{Cfloat},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     B::Ptr{Cfloat}, ldb::rocblas_int,
                                                     stride_b::rocblas_stride,
                                                     beta::Ptr{Cfloat}, C::Ptr{Cfloat},
                                                     ldc::rocblas_int,
                                                     stride_c::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_dgemmt_strided_batched(handle, uplo, transA, transB, n, k, alpha, A, lda,
                                        stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dgemmt_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     transA::rocblas_operation,
                                                     transB::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{Cdouble}, A::Ptr{Cdouble},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     B::Ptr{Cdouble}, ldb::rocblas_int,
                                                     stride_b::rocblas_stride,
                                                     beta::Ptr{Cdouble}, C::Ptr{Cdouble},
                                                     ldc::rocblas_int,
                                                     stride_c::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_cgemmt_strided_batched(handle, uplo, transA, transB, n, k, alpha, A, lda,
                                        stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_cgemmt_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     transA::rocblas_operation,
                                                     transB::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_float_complex},
                                                     A::Ptr{rocblas_float_complex},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     B::Ptr{rocblas_float_complex},
                                                     ldb::rocblas_int,
                                                     stride_b::rocblas_stride,
                                                     beta::Ptr{rocblas_float_complex},
                                                     C::Ptr{rocblas_float_complex},
                                                     ldc::rocblas_int,
                                                     stride_c::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_zgemmt_strided_batched(handle, uplo, transA, transB, n, k, alpha, A, lda,
                                        stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                                        batch_count)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_zgemmt_strided_batched(handle::rocblas_handle,
                                                     uplo::rocblas_fill,
                                                     transA::rocblas_operation,
                                                     transB::rocblas_operation,
                                                     n::rocblas_int, k::rocblas_int,
                                                     alpha::Ptr{rocblas_double_complex},
                                                     A::Ptr{rocblas_double_complex},
                                                     lda::rocblas_int,
                                                     stride_a::rocblas_stride,
                                                     B::Ptr{rocblas_double_complex},
                                                     ldb::rocblas_int,
                                                     stride_b::rocblas_stride,
                                                     beta::Ptr{rocblas_double_complex},
                                                     C::Ptr{rocblas_double_complex},
                                                     ldc::rocblas_int,
                                                     stride_c::rocblas_stride,
                                                     batch_count::rocblas_int)::rocblas_status
end

function rocblas_geam_ex(handle, transA, transB, m, n, k, alpha, A, a_type, lda, B, b_type,
                         ldb, beta, C, c_type, ldc, D, d_type, ldd, compute_type,
                         geam_ex_op)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_geam_ex(handle::rocblas_handle, transA::rocblas_operation,
                                      transB::rocblas_operation, m::rocblas_int,
                                      n::rocblas_int, k::rocblas_int, alpha::Ptr{Cvoid},
                                      A::Ptr{Cvoid}, a_type::rocblas_datatype,
                                      lda::rocblas_int, B::Ptr{Cvoid},
                                      b_type::rocblas_datatype, ldb::rocblas_int,
                                      beta::Ptr{Cvoid}, C::Ptr{Cvoid},
                                      c_type::rocblas_datatype, ldc::rocblas_int,
                                      D::Ptr{Cvoid}, d_type::rocblas_datatype,
                                      ldd::rocblas_int, compute_type::rocblas_datatype,
                                      geam_ex_op::rocblas_geam_ex_operation)::rocblas_status
end

function rocblas_trsm_batched_ex(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                                 ldb, batch_count, invA, invA_size, compute_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_trsm_batched_ex(handle::rocblas_handle, side::rocblas_side,
                                              uplo::rocblas_fill, transA::rocblas_operation,
                                              diag::rocblas_diagonal, m::rocblas_int,
                                              n::rocblas_int, alpha::Ptr{Cvoid},
                                              A::Ptr{Cvoid}, lda::rocblas_int,
                                              B::Ptr{Cvoid}, ldb::rocblas_int,
                                              batch_count::rocblas_int, invA::Ptr{Cvoid},
                                              invA_size::rocblas_int,
                                              compute_type::rocblas_datatype)::rocblas_status
end

function rocblas_trsm_strided_batched_ex(handle, side, uplo, transA, diag, m, n, alpha, A,
                                         lda, stride_A, B, ldb, stride_B, batch_count, invA,
                                         invA_size, stride_invA, compute_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_trsm_strided_batched_ex(handle::rocblas_handle,
                                                      side::rocblas_side,
                                                      uplo::rocblas_fill,
                                                      transA::rocblas_operation,
                                                      diag::rocblas_diagonal,
                                                      m::rocblas_int, n::rocblas_int,
                                                      alpha::Ptr{Cvoid}, A::Ptr{Cvoid},
                                                      lda::rocblas_int,
                                                      stride_A::rocblas_stride,
                                                      B::Ptr{Cvoid}, ldb::rocblas_int,
                                                      stride_B::rocblas_stride,
                                                      batch_count::rocblas_int,
                                                      invA::Ptr{Cvoid},
                                                      invA_size::rocblas_int,
                                                      stride_invA::rocblas_stride,
                                                      compute_type::rocblas_datatype)::rocblas_status
end

function rocblas_axpy_ex(handle, n, alpha, alpha_type, x, x_type, incx, y, y_type, incy,
                         execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_axpy_ex(handle::rocblas_handle, n::rocblas_int,
                                      alpha::Ptr{Cvoid}, alpha_type::rocblas_datatype,
                                      x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                      incx::rocblas_int, y::Ptr{Cvoid},
                                      y_type::rocblas_datatype, incy::rocblas_int,
                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_axpy_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, y, y_type,
                                 incy, batch_count, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_axpy_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                              alpha::Ptr{Cvoid},
                                              alpha_type::rocblas_datatype, x::Ptr{Cvoid},
                                              x_type::rocblas_datatype, incx::rocblas_int,
                                              y::Ptr{Cvoid}, y_type::rocblas_datatype,
                                              incy::rocblas_int, batch_count::rocblas_int,
                                              execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_axpy_strided_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx,
                                         stridex, y, y_type, incy, stridey, batch_count,
                                         execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_axpy_strided_batched_ex(handle::rocblas_handle,
                                                      n::rocblas_int, alpha::Ptr{Cvoid},
                                                      alpha_type::rocblas_datatype,
                                                      x::Ptr{Cvoid},
                                                      x_type::rocblas_datatype,
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      y::Ptr{Cvoid},
                                                      y_type::rocblas_datatype,
                                                      incy::rocblas_int,
                                                      stridey::rocblas_stride,
                                                      batch_count::rocblas_int,
                                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dot_ex(handle, n, x, x_type, incx, y, y_type, incy, result, result_type,
                        execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dot_ex(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cvoid},
                                     x_type::rocblas_datatype, incx::rocblas_int,
                                     y::Ptr{Cvoid}, y_type::rocblas_datatype,
                                     incy::rocblas_int, result::Ptr{Cvoid},
                                     result_type::rocblas_datatype,
                                     execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dotc_ex(handle, n, x, x_type, incx, y, y_type, incy, result, result_type,
                         execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dotc_ex(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cvoid},
                                      x_type::rocblas_datatype, incx::rocblas_int,
                                      y::Ptr{Cvoid}, y_type::rocblas_datatype,
                                      incy::rocblas_int, result::Ptr{Cvoid},
                                      result_type::rocblas_datatype,
                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dot_batched_ex(handle, n, x, x_type, incx, y, y_type, incy, batch_count,
                                result, result_type, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dot_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                             incx::rocblas_int, y::Ptr{Cvoid},
                                             y_type::rocblas_datatype, incy::rocblas_int,
                                             batch_count::rocblas_int, result::Ptr{Cvoid},
                                             result_type::rocblas_datatype,
                                             execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dotc_batched_ex(handle, n, x, x_type, incx, y, y_type, incy, batch_count,
                                 result, result_type, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dotc_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                              x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                              incx::rocblas_int, y::Ptr{Cvoid},
                                              y_type::rocblas_datatype, incy::rocblas_int,
                                              batch_count::rocblas_int, result::Ptr{Cvoid},
                                              result_type::rocblas_datatype,
                                              execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dot_strided_batched_ex(handle, n, x, x_type, incx, stride_x, y, y_type,
                                        incy, stride_y, batch_count, result, result_type,
                                        execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dot_strided_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cvoid},
                                                     x_type::rocblas_datatype,
                                                     incx::rocblas_int,
                                                     stride_x::rocblas_stride,
                                                     y::Ptr{Cvoid},
                                                     y_type::rocblas_datatype,
                                                     incy::rocblas_int,
                                                     stride_y::rocblas_stride,
                                                     batch_count::rocblas_int,
                                                     result::Ptr{Cvoid},
                                                     result_type::rocblas_datatype,
                                                     execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_dotc_strided_batched_ex(handle, n, x, x_type, incx, stride_x, y, y_type,
                                         incy, stride_y, batch_count, result, result_type,
                                         execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_dotc_strided_batched_ex(handle::rocblas_handle,
                                                      n::rocblas_int, x::Ptr{Cvoid},
                                                      x_type::rocblas_datatype,
                                                      incx::rocblas_int,
                                                      stride_x::rocblas_stride,
                                                      y::Ptr{Cvoid},
                                                      y_type::rocblas_datatype,
                                                      incy::rocblas_int,
                                                      stride_y::rocblas_stride,
                                                      batch_count::rocblas_int,
                                                      result::Ptr{Cvoid},
                                                      result_type::rocblas_datatype,
                                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_nrm2_ex(handle, n, x, x_type, incx, results, result_type, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_nrm2_ex(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cvoid},
                                      x_type::rocblas_datatype, incx::rocblas_int,
                                      results::Ptr{Cvoid}, result_type::rocblas_datatype,
                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_nrm2_batched_ex(handle, n, x, x_type, incx, batch_count, results,
                                 result_type, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_nrm2_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                              x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                              incx::rocblas_int, batch_count::rocblas_int,
                                              results::Ptr{Cvoid},
                                              result_type::rocblas_datatype,
                                              execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_nrm2_strided_batched_ex(handle, n, x, x_type, incx, stride_x, batch_count,
                                         results, result_type, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_nrm2_strided_batched_ex(handle::rocblas_handle,
                                                      n::rocblas_int, x::Ptr{Cvoid},
                                                      x_type::rocblas_datatype,
                                                      incx::rocblas_int,
                                                      stride_x::rocblas_stride,
                                                      batch_count::rocblas_int,
                                                      results::Ptr{Cvoid},
                                                      result_type::rocblas_datatype,
                                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_rot_ex(handle, n, x, x_type, incx, y, y_type, incy, c, s, cs_type,
                        execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_rot_ex(handle::rocblas_handle, n::rocblas_int, x::Ptr{Cvoid},
                                     x_type::rocblas_datatype, incx::rocblas_int,
                                     y::Ptr{Cvoid}, y_type::rocblas_datatype,
                                     incy::rocblas_int, c::Ptr{Cvoid}, s::Ptr{Cvoid},
                                     cs_type::rocblas_datatype,
                                     execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_rot_batched_ex(handle, n, x, x_type, incx, y, y_type, incy, c, s, cs_type,
                                batch_count, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_rot_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                             x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                             incx::rocblas_int, y::Ptr{Cvoid},
                                             y_type::rocblas_datatype, incy::rocblas_int,
                                             c::Ptr{Cvoid}, s::Ptr{Cvoid},
                                             cs_type::rocblas_datatype,
                                             batch_count::rocblas_int,
                                             execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_rot_strided_batched_ex(handle, n, x, x_type, incx, stride_x, y, y_type,
                                        incy, stride_y, c, s, cs_type, batch_count,
                                        execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_rot_strided_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                                     x::Ptr{Cvoid},
                                                     x_type::rocblas_datatype,
                                                     incx::rocblas_int,
                                                     stride_x::rocblas_stride,
                                                     y::Ptr{Cvoid},
                                                     y_type::rocblas_datatype,
                                                     incy::rocblas_int,
                                                     stride_y::rocblas_stride,
                                                     c::Ptr{Cvoid}, s::Ptr{Cvoid},
                                                     cs_type::rocblas_datatype,
                                                     batch_count::rocblas_int,
                                                     execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_scal_ex(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scal_ex(handle::rocblas_handle, n::rocblas_int,
                                      alpha::Ptr{Cvoid}, alpha_type::rocblas_datatype,
                                      x::Ptr{Cvoid}, x_type::rocblas_datatype,
                                      incx::rocblas_int,
                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_scal_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, batch_count,
                                 execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scal_batched_ex(handle::rocblas_handle, n::rocblas_int,
                                              alpha::Ptr{Cvoid},
                                              alpha_type::rocblas_datatype, x::Ptr{Cvoid},
                                              x_type::rocblas_datatype, incx::rocblas_int,
                                              batch_count::rocblas_int,
                                              execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_scal_strided_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx,
                                         stridex, batch_count, execution_type)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_scal_strided_batched_ex(handle::rocblas_handle,
                                                      n::rocblas_int, alpha::Ptr{Cvoid},
                                                      alpha_type::rocblas_datatype,
                                                      x::Ptr{Cvoid},
                                                      x_type::rocblas_datatype,
                                                      incx::rocblas_int,
                                                      stridex::rocblas_stride,
                                                      batch_count::rocblas_int,
                                                      execution_type::rocblas_datatype)::rocblas_status
end

function rocblas_status_to_string(status)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_status_to_string(status::rocblas_status)::Ptr{Cchar}
end

function rocblas_initialize()
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_initialize()::Cvoid
end

function rocblas_get_version_string(buf, len)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_version_string(buf::Ptr{Cchar},
                                                 len::Csize_t)::rocblas_status
end

function rocblas_get_version_string_size(len)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_version_string_size(len::Ptr{Csize_t})::rocblas_status
end

function rocblas_start_device_memory_size_query(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_start_device_memory_size_query(handle::rocblas_handle)::rocblas_status
end

function rocblas_stop_device_memory_size_query(handle, size)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_stop_device_memory_size_query(handle::rocblas_handle,
                                                            size::Ptr{Csize_t})::rocblas_status
end

function rocblas_is_device_memory_size_query(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_is_device_memory_size_query(handle::rocblas_handle)::Bool
end

function rocblas_device_malloc_success(ptr)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_device_malloc_success(ptr::Ptr{rocblas_device_malloc_base})::Bool
end

function rocblas_device_malloc_ptr(ptr, res)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_device_malloc_ptr(ptr::Ptr{rocblas_device_malloc_base},
                                                res::Ptr{Ptr{Cvoid}})::rocblas_status
end

function rocblas_device_malloc_get(ptr, index, res)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_device_malloc_get(ptr::Ptr{rocblas_device_malloc_base},
                                                index::Csize_t,
                                                res::Ptr{Ptr{Cvoid}})::rocblas_status
end

function rocblas_device_malloc_free(ptr)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_device_malloc_free(ptr::Ptr{rocblas_device_malloc_base})::rocblas_status
end

function rocblas_device_malloc_set_default_memory_size(size)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_device_malloc_set_default_memory_size(size::Csize_t)::Cvoid
end

function rocblas_get_device_memory_size(handle, size)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_get_device_memory_size(handle::rocblas_handle,
                                                     size::Ptr{Csize_t})::rocblas_status
end

function rocblas_set_device_memory_size(handle, size)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_device_memory_size(handle::rocblas_handle,
                                                     size::Csize_t)::rocblas_status
end

function rocblas_set_workspace(handle, addr, size)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_set_workspace(handle::rocblas_handle, addr::Ptr{Cvoid},
                                            size::Csize_t)::rocblas_status
end

function rocblas_is_managing_device_memory(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_is_managing_device_memory(handle::rocblas_handle)::Bool
end

function rocblas_is_user_managing_device_memory(handle)
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_is_user_managing_device_memory(handle::rocblas_handle)::Bool
end

function rocblas_abort()
    AMDGPU.prepare_state()
    @ccall librocblas.rocblas_abort()::Cvoid
end
