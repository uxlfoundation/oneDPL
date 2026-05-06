// -*- C++ -*-
//===-- rts_bridge_mat_3l_standalone.pass.cpp --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3l (Matrix2x2): STANDALONE reproducer for the Windows CPU
// release-mode segfault. ZERO oneDPL includes - only SYCL + standard library.
//
// The bug is triggered by the combination of:
//   - [[sycl::reqd_sub_group_size(32)]] on the kernel lambda
//   - body<bool>() member function template on the submitter struct
//   - [=, *this] capture in the lambda
//   - Dual-path instantiation: if(!slm) body<true>() else body<false>()
//   - Matrix2x2<int32_t> as the value type (16 bytes, non-commutative)

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <utility>

// ============================================================================
// Inlined types from oneDPL
// ============================================================================

template <typename T>
struct Matrix2x2 {
    T a00, a01, a10, a11;

    Matrix2x2() : a00(1), a01(0), a10(0), a11(1) {} // identity
    Matrix2x2(T x, T y)
        : a00(x), a01(x + 1), a10(y), a11(y + 1) {}
    Matrix2x2(T _a00, T _a01, T _a10, T _a11)
        : a00(_a00), a01(_a01), a10(_a10), a11(_a11) {}

    bool operator==(const Matrix2x2& o) const {
        return a00 == o.a00 && a01 == o.a01 && a10 == o.a10 && a11 == o.a11;
    }
    bool operator!=(const Matrix2x2& o) const { return !(*this == o); }
};

template <typename T>
struct multiply_matrix {
    Matrix2x2<T> operator()(const Matrix2x2<T>& a, const Matrix2x2<T>& b) const {
        return Matrix2x2<T>(
            a.a00 * b.a00 + a.a01 * b.a10,
            a.a00 * b.a01 + a.a01 * b.a11,
            a.a10 * b.a00 + a.a11 * b.a10,
            a.a10 * b.a01 + a.a11 * b.a11);
    }
};

// __lazy_ctor_storage
template <typename _Tp>
union __lazy_ctor_storage {
    using __value_type = _Tp;
    _Tp __v;
    __lazy_ctor_storage() {}
    ~__lazy_ctor_storage() {}

    template <typename _U>
    void __setup(_U&& init) { new (&__v) _Tp(std::forward<_U>(init)); }
    void __destroy() { __v.~_Tp(); }
};

// __no_init_value
template <typename _InitType>
struct __no_init_value {
    using __value_type = _InitType;
};

// __init_processing (no-op for __no_init_value)
template <typename _InitType>
struct __init_processing {
    template <typename _Tp>
    void operator()(const __no_init_value<_InitType>&, _Tp&&) const {}
    template <typename _Tp, typename _BinaryOp>
    void operator()(const __no_init_value<_InitType>&, _Tp&&, _BinaryOp) const {}
};

// identity functor
struct identity_fn {
    template <typename T>
    T&& operator()(T&& x) const { return std::forward<T>(x); }
};

// __noop_temp_data
struct __noop_temp_data {
    template <typename _ValueT>
    void set(std::uint16_t, const _ValueT&) const {}
};

// __gen_transform_input
template <typename _UnaryOp, typename _InitType>
struct __gen_transform_input {
    using TempData = __noop_temp_data;
    template <typename _InRng>
    auto operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const {
        return static_cast<_InitType>(__unary_op(__in_rng[__id]));
    }
    _UnaryOp __unary_op;
};

// __simple_write_to_id
struct __simple_write_to_id {
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _ValueType>
    void operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v, const _TempData&) const {
        __out_rng[__id] = __v;
    }
};

// ============================================================================
// Sub-group communication primitives
// ============================================================================

template <bool __use_subgroup_ops, typename _ValueType>
_ValueType
__shift_group_right(const sycl::sub_group& __sub_group, _ValueType __value, std::uint32_t __shift,
                    _ValueType* __comm_slm)
{
    if constexpr (__use_subgroup_ops) {
        return sycl::shift_group_right(__sub_group, __value, __shift);
    } else {
        std::uint32_t __local_id = __sub_group.get_local_linear_id();
        std::uint32_t __sg_base = __sub_group.get_group_linear_id() * __sub_group.get_max_local_range()[0];
        __comm_slm[__sg_base + __local_id] = __value;
        sycl::group_barrier(__sub_group);
        _ValueType __result = __comm_slm[__sg_base + ((__local_id >= __shift) ? __local_id - __shift : __local_id)];
        sycl::group_barrier(__sub_group);
        return __result;
    }
}

template <bool __use_subgroup_ops, typename _ValueType, typename _IdType>
_ValueType
__group_broadcast(const sycl::sub_group& __sub_group, _ValueType __value, _IdType __broadcast_id,
                  _ValueType* __comm_slm)
{
    if constexpr (__use_subgroup_ops) {
        return sycl::group_broadcast(__sub_group, __value, __broadcast_id);
    } else {
        std::uint32_t __local_id = __sub_group.get_local_linear_id();
        std::uint32_t __sg_base = __sub_group.get_group_linear_id() * __sub_group.get_max_local_range()[0];
        __comm_slm[__sg_base + __local_id] = __value;
        sycl::group_barrier(__sub_group);
        _ValueType __result = __comm_slm[__sg_base + __broadcast_id];
        sycl::group_barrier(__sub_group);
        return __result;
    }
}

// ============================================================================
// Sub-group scan primitives
// ============================================================================

template <bool __use_subgroup_ops, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__inclusive_sub_group_masked_scan(const sycl::sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value,
                                  _BinaryOp __binary_op, _LazyValueType& __init_and_carry,
                                  _ValueType* __comm_slm)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    const std::uint8_t __sub_group_size = __sub_group.get_max_local_range()[0];
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1) {
        _ValueType __partial_carry_in =
            __shift_group_right<__use_subgroup_ops>(__sub_group, __value, __shift, __comm_slm);
        if (__mask_fn(__sub_group_local_id, __shift))
            __value = __binary_op(__partial_carry_in, __value);
    }
    if constexpr (__init_present) {
        __value = __binary_op(__init_and_carry.__v, __value);
        __init_and_carry.__v =
            __group_broadcast<__use_subgroup_ops>(__sub_group, __value, __init_broadcast_id, __comm_slm);
    } else {
        __init_and_carry.__setup(
            __group_broadcast<__use_subgroup_ops>(__sub_group, __value, __init_broadcast_id, __comm_slm));
    }
}

template <bool __use_subgroup_ops, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_masked_scan(const sycl::sub_group& __sub_group, _MaskOp __mask_fn,
                        _InitBroadcastId __init_broadcast_id, _ValueType& __value,
                        _BinaryOp __binary_op, _LazyValueType& __init_and_carry,
                        _ValueType* __comm_slm)
{
    // We only need the inclusive path for this test
    __inclusive_sub_group_masked_scan<__use_subgroup_ops, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry, __comm_slm);
}

template <bool __use_subgroup_ops, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType>
void
__sub_group_scan(const sycl::sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                 _LazyValueType& __init_and_carry, _ValueType* __comm_slm)
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset;
    };
    std::uint8_t __init_broadcast_id = __sub_group.get_max_local_range()[0] - 1;
    __sub_group_masked_scan<__use_subgroup_ops, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry, __comm_slm);
}

template <bool __use_subgroup_ops, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType, typename _SizeType>
void
__sub_group_scan_partial(const sycl::sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                         _LazyValueType& __init_and_carry, _SizeType __elements_to_process,
                         _ValueType* __comm_slm)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__use_subgroup_ops, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry, __comm_slm);
}

// ============================================================================
// __scan_through_elements_helper
// ============================================================================

template <bool __use_subgroup_ops, bool __is_inclusive, bool __init_present, bool __capture_output,
          std::uint16_t __max_inputs_per_item, typename _GenInput, typename _ScanInputTransform,
          typename _BinaryOp, typename _WriteOp, typename _LazyValueType, typename _InRng,
          typename _OutRng, typename _ScanValueType>
void
__scan_through_elements_helper(const sycl::sub_group& __sub_group, _GenInput __gen_input,
                               _ScanInputTransform __scan_input_transform, _BinaryOp __binary_op,
                               _WriteOp __write_op, _LazyValueType& __sub_group_carry,
                               const _InRng& __in_rng, _OutRng& __out_rng,
                               std::size_t __start_id, std::size_t __n, std::uint32_t __iters_per_item,
                               std::size_t __subgroup_start_id, std::uint32_t __sub_group_id,
                               std::uint32_t __active_subgroups, _ScanValueType* __comm_slm)
{
    using _GenInputType = std::invoke_result_t<_GenInput, _InRng, std::size_t, typename _GenInput::TempData&>;
    const std::uint8_t __sub_group_size = __sub_group.get_max_local_range()[0];
    bool __is_full_block = (__iters_per_item == __max_inputs_per_item);
    bool __is_full_thread = __subgroup_start_id + __iters_per_item * __sub_group_size <= __n;
    using _TempData = typename _GenInput::TempData;
    _TempData __temp_data{};

    if (__is_full_thread) {
        _GenInputType __v = __gen_input(__in_rng, __start_id, __temp_data);
        __sub_group_scan<__use_subgroup_ops, __is_inclusive, __init_present>(
            __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry, __comm_slm);
        if constexpr (__capture_output)
            __write_op(__out_rng, __start_id, __v, __temp_data);

        if (__is_full_block) {
            for (std::uint32_t __j = 1; __j < __max_inputs_per_item; __j++) {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size, __temp_data);
                __sub_group_scan<__use_subgroup_ops, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry, __comm_slm);
                if constexpr (__capture_output)
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v, __temp_data);
            }
        } else {
            for (std::uint32_t __j = 1; __j < __iters_per_item; __j++) {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size, __temp_data);
                __sub_group_scan<__use_subgroup_ops, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry, __comm_slm);
                if constexpr (__capture_output)
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v, __temp_data);
            }
        }
    } else {
        if (__sub_group_id < __active_subgroups) {
            auto __ceiling_div = [](auto __a, auto __b) { return (__a + __b - 1) / __b; };
            std::uint32_t __iters = __ceiling_div(__n - __subgroup_start_id, (std::size_t)__sub_group_size);

            if (__iters == 1) {
                std::size_t __local_id = (__start_id < __n) ? __start_id : __n - 1;
                _GenInputType __v = __gen_input(__in_rng, __local_id, __temp_data);
                __sub_group_scan_partial<__use_subgroup_ops, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - __subgroup_start_id, __comm_slm);
                if constexpr (__capture_output) {
                    if (__start_id < __n)
                        __write_op(__out_rng, __start_id, __v, __temp_data);
                }
            } else {
                _GenInputType __v = __gen_input(__in_rng, __start_id, __temp_data);
                __sub_group_scan<__use_subgroup_ops, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry, __comm_slm);
                if constexpr (__capture_output)
                    __write_op(__out_rng, __start_id, __v, __temp_data);

                for (std::uint32_t __j = 1; __j < __iters - 1; __j++) {
                    std::size_t __local_id = __start_id + __j * __sub_group_size;
                    __v = __gen_input(__in_rng, __local_id, __temp_data);
                    __sub_group_scan<__use_subgroup_ops, __is_inclusive, /*__init_present=*/true>(
                        __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry, __comm_slm);
                    if constexpr (__capture_output)
                        __write_op(__out_rng, __local_id, __v, __temp_data);
                }

                std::size_t __offset = __start_id + (__iters - 1) * __sub_group_size;
                std::size_t __local_id = (__offset < __n) ? __offset : __n - 1;
                __v = __gen_input(__in_rng, __local_id, __temp_data);
                __sub_group_scan_partial<__use_subgroup_ops, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - (__subgroup_start_id + (__iters - 1) * __sub_group_size), __comm_slm);
                if constexpr (__capture_output) {
                    if (__offset < __n)
                        __write_op(__out_rng, __offset, __v, __temp_data);
                }
            }
        }
    }
}

// ============================================================================
// __reduce_then_scan_sub_group_params
// ============================================================================

struct __reduce_then_scan_sub_group_params {
    __reduce_then_scan_sub_group_params(std::uint32_t __work_group_size, std::uint8_t __sub_group_size,
                                        std::uint32_t __max_num_work_groups, std::uint32_t __max_block_size,
                                        std::size_t __inputs_remaining)
        : __num_sub_groups_local(__work_group_size / __sub_group_size),
          __num_sub_groups_global(__num_sub_groups_local * __max_num_work_groups)
    {
        const std::uint32_t __max_inputs_per_subgroup = __max_block_size / __num_sub_groups_global;
        auto __bit_ceil = [](std::size_t v) -> std::size_t {
            if (v <= 1) return 1;
            std::size_t p = 1;
            while (p < v) p <<= 1;
            return p;
        };
        const std::uint32_t __evenly_divided_remaining_inputs =
            std::max(std::size_t{__sub_group_size}, __bit_ceil(__inputs_remaining) / __num_sub_groups_global);
        __inputs_per_sub_group =
            __inputs_remaining >= __max_block_size ? __max_inputs_per_subgroup : __evenly_divided_remaining_inputs;
        __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
    }

    std::uint32_t __num_sub_groups_local;
    std::uint32_t __num_sub_groups_global;
    std::uint32_t __inputs_per_sub_group;
    std::uint32_t __inputs_per_item;
};

// ============================================================================
// Buffer-backed range wrappers
// ============================================================================

template <typename T, sycl::access::mode M>
struct buf_range {
    sycl::accessor<T, 1, M, sycl::target::device> __acc;

    T operator[](std::size_t __id) const { return __acc[__id]; }

    template <sycl::access::mode M2 = M>
    std::enable_if_t<M2 == sycl::access::mode::read_write, T&>
    operator[](std::size_t __id) { return __acc[__id]; }
};

// Wrapper to pass buffer reference for deferred accessor creation inside submit
template <typename T>
struct buf_ref {
    sycl::buffer<T, 1>& __buf;
    std::size_t __n;
};

// ============================================================================
// Scratch storage (simplified buffer-based)
// ============================================================================

template <typename T>
struct scratch_storage {
    sycl::buffer<T, 1> __buf;
    std::size_t __scratch_n;

    scratch_storage(std::size_t scratch_n)
        : __buf(sycl::range<1>(scratch_n + 2)), __scratch_n(scratch_n) {}

    // For reduce kernel: write-only scratch accessor
    auto get_scratch_acc_write(sycl::handler& cgh) {
        return __buf.template get_access<sycl::access::mode::write>(cgh);
    }
    // For scan kernel: read-write scratch accessor
    auto get_scratch_acc_rw(sycl::handler& cgh) {
        return __buf.template get_access<sycl::access::mode::read_write>(cgh);
    }
    // Result accessor (last 2 elements)
    auto get_result_acc_write(sycl::handler& cgh) {
        return __buf.template get_access<sycl::access::mode::write>(cgh);
    }

    // Static helper to get raw pointer from accessor
    template <typename Acc>
    static T* get_ptr(const Acc& acc) {
        return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
    }
};

// ============================================================================
// Utility
// ============================================================================

inline std::uint32_t __dpl_ceiling_div(std::size_t a, std::size_t b) {
    return static_cast<std::uint32_t>((a + b - 1) / b);
}

inline std::size_t __dpl_bit_ceil(std::size_t v) {
    if (v <= 1) return 1;
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ============================================================================
// REDUCE SUBMITTER
// ============================================================================

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, typename _GenReduceInput,
          typename _ReduceOp, typename _InitType>
struct reduce_submitter {
    using _InitValueType = typename _InitType::__value_type;

    // *** THE CRITICAL PATTERN: member function template called from kernel lambda ***
    template <bool __use_subgroup_ops, typename _TmpAcc, typename _InRng>
    void
    __reduce_kernel_body(sycl::nd_item<1> __ndi,
                         sycl::local_accessor<_InitValueType, 1> __sub_group_partials,
                         sycl::local_accessor<_InitValueType, 1> __comm_slm,
                         _TmpAcc __tmp_acc, _InRng __in_rng,
                         const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        sycl::sub_group __sub_group = __ndi.get_sub_group();
        const std::uint8_t __sub_group_size = __sub_group.get_max_local_range()[0];

        __reduce_then_scan_sub_group_params __sub_group_params(
            __work_group_size, __sub_group_size, __max_num_work_groups, __max_block_size, __inputs_remaining);

        _InitValueType* __comm_slm_ptr = __use_subgroup_ops ? nullptr : &__comm_slm[0];
        std::size_t __group_id = __ndi.get_group(0);
        std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
        std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

        __lazy_ctor_storage<_InitValueType> __sub_group_carry;
        std::size_t __group_start_id =
            (__block_num * __max_block_size) +
            (__group_id * __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local);

        std::size_t __max_inputs_in_group =
            __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local;
        std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
        std::uint32_t __active_subgroups =
            __dpl_ceiling_div(__inputs_in_group, __sub_group_params.__inputs_per_sub_group);
        std::size_t __subgroup_start_id =
            __group_start_id + (__sub_group_id * __sub_group_params.__inputs_per_sub_group);
        std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

        if (__sub_group_id < __active_subgroups) {
            __scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                                           /*__init_present=*/false,
                                           /*__capture_output=*/false, __max_inputs_per_item>(
                __sub_group, __gen_reduce_input, identity_fn{}, __reduce_op, nullptr, __sub_group_carry,
                __in_rng, /*unused*/ __in_rng, __start_id, __n, __sub_group_params.__inputs_per_item,
                __subgroup_start_id, __sub_group_id, __active_subgroups, __comm_slm_ptr);
            if (__sub_group_local_id == 0)
                __sub_group_partials[__sub_group_id] = __sub_group_carry.__v;
            __sub_group_carry.__destroy();
        }
        sycl::group_barrier(__ndi.get_group());

        // Compute prefix on sub-group carries and write to scratch
        if (__sub_group_id == 0) {
            __start_id = (__group_id * __sub_group_params.__num_sub_groups_local);
            std::uint8_t __iters = __dpl_ceiling_div(__active_subgroups, __sub_group_size);

            if (__iters == 1) {
                std::uint32_t __load_id = std::min(std::uint32_t{__sub_group_local_id}, __active_subgroups - 1);
                _InitValueType __v = __sub_group_partials[__load_id];
                __sub_group_scan_partial<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/false>(
                    __sub_group, __v, __reduce_op, __sub_group_carry, __active_subgroups, __comm_slm_ptr);
                if (__sub_group_local_id < __active_subgroups)
                    __tmp_acc[__start_id + __sub_group_local_id] = __v;
            } else {
                std::uint32_t __reduction_scan_id = __sub_group_local_id;
                _InitValueType __v = __sub_group_partials[__reduction_scan_id];
                __sub_group_scan<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/false>(
                    __sub_group, __v, __reduce_op, __sub_group_carry, __comm_slm_ptr);
                __tmp_acc[__start_id + __reduction_scan_id] = __v;
                __reduction_scan_id += __sub_group_size;

                for (std::uint32_t __i = 1; __i < __iters - 1; __i++) {
                    __v = __sub_group_partials[__reduction_scan_id];
                    __sub_group_scan<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/true>(
                        __sub_group, __v, __reduce_op, __sub_group_carry, __comm_slm_ptr);
                    __tmp_acc[__start_id + __reduction_scan_id] = __v;
                    __reduction_scan_id += __sub_group_size;
                }

                std::uint32_t __load_id = std::min(__reduction_scan_id, __sub_group_params.__num_sub_groups_local - 1);
                __v = __sub_group_partials[__load_id];
                __sub_group_scan_partial<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/true>(
                    __sub_group, __v, __reduce_op, __sub_group_carry,
                    __active_subgroups - ((__iters - 1) * __sub_group_size), __comm_slm_ptr);
                if (__reduction_scan_id < __sub_group_params.__num_sub_groups_local)
                    __tmp_acc[__start_id + __reduction_scan_id] = __v;
            }
            __sub_group_carry.__destroy();
        }
    }

    // operator() - THE CRITICAL PATTERN: [=, *this] + reqd_sub_group_size(32) + dual path
    template <typename _ScratchStorage>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range, buf_ref<_InitValueType> __in_ref,
               _ScratchStorage& __scratch, const sycl::event& __prior_event,
               const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        return __q.submit([&, this](sycl::handler& __cgh) {
            sycl::local_accessor<_InitValueType, 1> __sub_group_partials(__max_num_sub_groups_local, __cgh);
            sycl::local_accessor<_InitValueType, 1> __comm_slm(__use_slm_for_comm ? __work_group_size : 0, __cgh);
            __cgh.depends_on(__prior_event);
            auto __temp_acc = __scratch.get_scratch_acc_write(__cgh);
            auto __in_acc = __in_ref.__buf.template get_access<sycl::access::mode::read>(__cgh);

            buf_range<_InitValueType, sycl::access::mode::read> __in_range{__in_acc};

            __cgh.parallel_for(
                __nd_range, [=, *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(32)]] {
                    _InitValueType* __tmp_ptr = scratch_storage<_InitValueType>::get_ptr(__temp_acc);

                    // *** DUAL PATH INSTANTIATION - critical to reproducing the bug ***
                    if (!__use_slm_for_comm)
                        __reduce_kernel_body<true>(__ndi, __sub_group_partials, __comm_slm, __tmp_ptr,
                                                   __in_range, __inputs_remaining, __block_num);
                    else
                        __reduce_kernel_body<false>(__ndi, __sub_group_partials, __comm_slm, __tmp_ptr,
                                                    __in_range, __inputs_remaining, __block_num);
                });
        });
    }

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::size_t __n;
    const bool __use_slm_for_comm;

    const _GenReduceInput __gen_reduce_input;
    const _ReduceOp __reduce_op;
    _InitType __init;
};

// ============================================================================
// SCAN SUBMITTER
// ============================================================================

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType>
struct scan_submitter {
    using _InitValueType = typename _InitType::__value_type;

    template <typename _TmpAcc>
    _InitValueType
    __get_block_carry_in(const std::size_t __block_num, _TmpAcc __tmp_acc,
                         const std::size_t __num_sub_groups_global) const
    {
        return __tmp_acc[__num_sub_groups_global + (__block_num % 2)];
    }

    template <typename _TmpAcc, typename _ValueType>
    void
    __set_block_carry_out(const std::size_t __block_num, _TmpAcc __tmp_acc, const _ValueType __block_carry_out,
                          const std::size_t __num_sub_groups_global) const
    {
        __tmp_acc[__num_sub_groups_global + 1 - (__block_num % 2)] = __block_carry_out;
    }

    // *** THE CRITICAL PATTERN: member function template ***
    template <bool __use_subgroup_ops, typename _TmpAcc, typename _ResAcc, typename _InRng, typename _OutRng>
    void
    __scan_kernel_body(sycl::nd_item<1> __ndi,
                       sycl::local_accessor<_InitValueType, 1> __sub_group_partials,
                       sycl::local_accessor<_InitValueType, 1> __comm_slm,
                       _TmpAcc __tmp_acc, _ResAcc __res_acc,
                       _InRng __in_rng, _OutRng __out_rng, std::uint32_t __inputs_in_block,
                       const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        sycl::sub_group __sub_group = __ndi.get_sub_group();
        const std::uint8_t __sub_group_size = __sub_group.get_max_local_range()[0];
        _InitValueType* __comm_slm_ptr = __use_subgroup_ops ? nullptr : &__comm_slm[0];

        __reduce_then_scan_sub_group_params __sub_group_params(
            __work_group_size, __sub_group_size, __max_num_work_groups, __max_block_size, __inputs_remaining);

        const std::uint32_t __active_groups = __dpl_ceiling_div(
            __inputs_in_block, __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local);

        std::uint32_t __group_id = __ndi.get_group(0);
        std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
        std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

        std::size_t __group_start_id =
            (__block_num * __max_block_size) +
            (__group_id * __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local);

        std::size_t __max_inputs_in_group =
            __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local;
        std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
        std::uint32_t __active_subgroups =
            __dpl_ceiling_div(__inputs_in_group, __sub_group_params.__inputs_per_sub_group);
        __lazy_ctor_storage<_InitValueType> __carry_last;
        __lazy_ctor_storage<_InitValueType> __sub_group_carry;

        // Step 1-4: Load sub-group partials and compute global carries
        if (__sub_group_id == 0) {
            std::uint8_t __iters = __dpl_ceiling_div(__active_subgroups, __sub_group_size);
            std::size_t __subgroups_before_my_group = __group_id * __sub_group_params.__num_sub_groups_local;
            std::uint32_t __load_reduction_id = __sub_group_local_id;

            std::uint8_t __i = 0;
            for (; __i < __iters - 1; __i++) {
                __sub_group_partials[__load_reduction_id] =
                    __tmp_acc[__subgroups_before_my_group + __load_reduction_id];
                __load_reduction_id += __sub_group_size;
            }
            if (__load_reduction_id < __active_subgroups) {
                __sub_group_partials[__load_reduction_id] =
                    __tmp_acc[__subgroups_before_my_group + __load_reduction_id];
            }

            // Step 2: compute global prefix from work-group carries
            std::uint32_t __offset = __sub_group_params.__num_sub_groups_local - 1;
            if (__group_id > 0) {
                const std::size_t __elements_to_process =
                    __subgroups_before_my_group / __sub_group_params.__num_sub_groups_local;
                const std::size_t __pre_carry_iters =
                    __dpl_ceiling_div(__elements_to_process, __sub_group_size);

                if (__pre_carry_iters == 1) {
                    std::size_t __proposed_id =
                        __sub_group_params.__num_sub_groups_local * __sub_group_local_id + __offset;
                    std::size_t __reduction_id =
                        (__proposed_id < __subgroups_before_my_group) ? __proposed_id : __subgroups_before_my_group - 1;
                    _InitValueType __value = __tmp_acc[__reduction_id];
                    __sub_group_scan_partial<__use_subgroup_ops, /*__is_inclusive=*/true,
                                             /*__init_present=*/false>(
                        __sub_group, __value, __reduce_op, __carry_last, __elements_to_process, __comm_slm_ptr);
                } else {
                    std::uint32_t __reduction_id =
                        __sub_group_params.__num_sub_groups_local * __sub_group_local_id + __offset;
                    std::uint32_t __reduction_id_increment =
                        __sub_group_params.__num_sub_groups_local * __sub_group_size;
                    _InitValueType __value = __tmp_acc[__reduction_id];
                    __sub_group_scan<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/false>(
                        __sub_group, __value, __reduce_op, __carry_last, __comm_slm_ptr);
                    __reduction_id += __reduction_id_increment;

                    for (std::uint32_t __ii = 1; __ii < __pre_carry_iters - 1; __ii++) {
                        __value = __tmp_acc[__reduction_id];
                        __sub_group_scan<__use_subgroup_ops, /*__is_inclusive=*/true, /*__init_present=*/true>(
                            __sub_group, __value, __reduce_op, __carry_last, __comm_slm_ptr);
                        __reduction_id += __reduction_id_increment;
                    }

                    std::size_t __remaining_elements =
                        __elements_to_process - ((__pre_carry_iters - 1) * __sub_group_size);
                    std::size_t __final_reduction_id =
                        std::min(std::size_t{__reduction_id}, __subgroups_before_my_group - 1);
                    __value = __tmp_acc[__final_reduction_id];
                    __sub_group_scan_partial<__use_subgroup_ops, /*__is_inclusive=*/true,
                                             /*__init_present=*/true>(
                        __sub_group, __value, __reduce_op, __carry_last, __remaining_elements, __comm_slm_ptr);
                }

                // Steps 3+4: apply global carry to local sub-group partials
                std::size_t __carry_offset = __sub_group_local_id;
                std::uint8_t __iters2 = __dpl_ceiling_div(__active_subgroups, __sub_group_size);
                std::uint8_t __ii = 0;
                for (; __ii < __iters2 - 1; ++__ii) {
                    __sub_group_partials[__carry_offset] =
                        __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                    __carry_offset += __sub_group_size;
                }
                if (__ii * __sub_group_size + __sub_group_local_id < __active_subgroups) {
                    __sub_group_partials[__carry_offset] =
                        __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                }
                if (__sub_group_local_id == 0)
                    __sub_group_partials[__active_subgroups] = __carry_last.__v;
                __carry_last.__destroy();
            }
        }

        sycl::group_barrier(__ndi.get_group());

        // Get inter-work group and adjusted for intra-work group prefix
        bool __sub_group_carry_initialized = true;
        if (__block_num == 0) {
            if (__sub_group_id > 0) {
                _InitValueType __value = __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                __init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                __sub_group_carry.__setup(__value);
            } else if (__group_id > 0) {
                _InitValueType __value = __sub_group_partials[__active_subgroups];
                __init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                __sub_group_carry.__setup(__value);
            } else {
                // No init value case: no carry
                __sub_group_carry_initialized = false;
            }
        } else {
            if (__sub_group_id > 0) {
                _InitValueType __value = __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                __sub_group_carry.__setup(__reduce_op(
                    __get_block_carry_in(__block_num, __tmp_acc, __sub_group_params.__num_sub_groups_global), __value));
            } else if (__group_id > 0) {
                __sub_group_carry.__setup(__reduce_op(
                    __get_block_carry_in(__block_num, __tmp_acc, __sub_group_params.__num_sub_groups_global),
                    __sub_group_partials[__active_subgroups]));
            } else {
                __sub_group_carry.__setup(
                    __get_block_carry_in(__block_num, __tmp_acc, __sub_group_params.__num_sub_groups_global));
            }
        }

        // Step 5: apply global carries and scan through elements
        std::size_t __subgroup_start_id =
            __group_start_id + (__sub_group_id * __sub_group_params.__inputs_per_sub_group);
        std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

        if (__sub_group_carry_initialized) {
            __scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                                           /*__init_present=*/true,
                                           /*__capture_output=*/true, __max_inputs_per_item>(
                __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op, __sub_group_carry,
                __in_rng, __out_rng, __start_id, __n, __sub_group_params.__inputs_per_item, __subgroup_start_id,
                __sub_group_id, __active_subgroups, __comm_slm_ptr);
        } else {
            __scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                                           /*__init_present=*/false,
                                           /*__capture_output=*/true, __max_inputs_per_item>(
                __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op, __sub_group_carry,
                __in_rng, __out_rng, __start_id, __n, __sub_group_params.__inputs_per_item, __subgroup_start_id,
                __sub_group_id, __active_subgroups, __comm_slm_ptr);
        }

        // Write block carry out
        if (__sub_group_local_id == 0 && (__active_groups == __group_id + 1) &&
            (__active_subgroups == __sub_group_id + 1)) {
            if (__block_num + 1 == __num_blocks) {
                __res_acc[0] = __sub_group_carry.__v;
            } else {
                __set_block_carry_out(__block_num, __tmp_acc, __sub_group_carry.__v,
                                      __sub_group_params.__num_sub_groups_global);
            }
        }
        __sub_group_carry.__destroy();
    }

    // operator() - THE CRITICAL PATTERN: [=, *this] + reqd_sub_group_size(32) + dual path
    template <typename _ScratchStorage>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range,
               buf_ref<_InitValueType> __in_ref, buf_ref<_InitValueType> __out_ref,
               _ScratchStorage& __scratch, const sycl::event& __prior_event,
               const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        std::size_t __num_remaining = __n - __block_num * __max_block_size;
        std::uint32_t __inputs_in_block = std::min(__num_remaining, std::size_t{__max_block_size});

        return __q.submit([&, this, __inputs_in_block](sycl::handler& __cgh) {
            sycl::local_accessor<_InitValueType, 1> __sub_group_partials(__max_num_sub_groups_local + 1, __cgh);
            sycl::local_accessor<_InitValueType, 1> __comm_slm(__use_slm_for_comm ? __work_group_size : 0, __cgh);
            __cgh.depends_on(__prior_event);

            auto __temp_acc = __scratch.get_scratch_acc_rw(__cgh);
            auto __res_acc = __scratch.get_result_acc_write(__cgh);
            auto __in_acc = __in_ref.__buf.template get_access<sycl::access::mode::read>(__cgh);
            auto __out_acc = __out_ref.__buf.template get_access<sycl::access::mode::read_write>(__cgh);

            buf_range<_InitValueType, sycl::access::mode::read> __in_range{__in_acc};
            buf_range<_InitValueType, sycl::access::mode::read_write> __out_range{__out_acc};

            __cgh.parallel_for(
                __nd_range, [=, *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(32)]] {
                    _InitValueType* __tmp_ptr = scratch_storage<_InitValueType>::get_ptr(__temp_acc);
                    _InitValueType* __res_ptr = scratch_storage<_InitValueType>::get_ptr(__res_acc) +
                                                __max_num_sub_groups_global + 2;

                    // *** DUAL PATH INSTANTIATION - critical to reproducing the bug ***
                    if (!__use_slm_for_comm)
                        __scan_kernel_body<true>(__ndi, __sub_group_partials, __comm_slm, __tmp_ptr, __res_ptr,
                                                 __in_range, __out_range, __inputs_in_block,
                                                 __inputs_remaining, __block_num);
                    else
                        __scan_kernel_body<false>(__ndi, __sub_group_partials, __comm_slm, __tmp_ptr, __res_ptr,
                                                  __in_range, __out_range, __inputs_in_block,
                                                  __inputs_remaining, __block_num);
                });
        });
    }

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::uint32_t __max_num_sub_groups_global;
    const std::size_t __num_blocks;
    const std::size_t __n;
    const bool __use_slm_for_comm;

    const _ReduceOp __reduce_op;
    const _GenScanInput __gen_scan_input;
    const _ScanInputTransform __scan_input_transform;
    const _WriteOp __write_op;
    _InitType __init;
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    using T = Matrix2x2<std::int32_t>;
    using _UnaryOp = identity_fn;
    using _BinaryOp = multiply_matrix<std::int32_t>;
    using _InitType = __no_init_value<T>;
    using _GenInput = __gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = identity_fn;
    using _WriteOp = __simple_write_to_id;

    constexpr std::size_t N = 20000;
    constexpr std::uint16_t __max_inputs_per_item =
        std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});
    constexpr bool __inclusive = true;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);

    _BinaryOp mat_op{};
    std::vector<T> h_expected(N);
    h_expected[0] = h_input[0];
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = mat_op(h_expected[i - 1], h_input[i]);

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[standalone_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Parameter calculation (mirrors oneDPL)
    const auto __supported_sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    const std::uint8_t __min_sub_group_size =
        *std::min_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());
    const std::uint8_t __max_sub_group_size =
        *std::max_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());

    const std::uint32_t __wg_size_cap = dev.is_gpu() ? 1024 : 256;
    // Query max work group size, cap at __wg_size_cap
    const std::uint32_t __max_work_group_size = std::min(
        __wg_size_cap,
        (std::uint32_t)dev.get_info<sycl::info::device::max_work_group_size>());
    const std::uint32_t __work_group_size = (__max_work_group_size / __max_sub_group_size) * __max_sub_group_size;

    const std::uint32_t __num_work_groups =
        __dpl_bit_ceil(dev.get_info<sycl::info::device::max_compute_units>()) /
        (dev.is_gpu() ? 4 : 1);

    const std::uint32_t __max_num_sub_groups_local = __work_group_size / __min_sub_group_size;
    const std::uint32_t __max_num_sub_groups_global = __max_num_sub_groups_local * __num_work_groups;
    const std::uint32_t __max_inputs_per_block = __work_group_size * __max_inputs_per_item * __num_work_groups;

    std::size_t __inputs_remaining = N;
    std::uint32_t __inputs_per_item =
        __inputs_remaining >= __max_inputs_per_block
            ? __max_inputs_per_item
            : __dpl_ceiling_div(__dpl_bit_ceil(__inputs_remaining),
                                __num_work_groups * __work_group_size);
    const std::size_t __block_size = std::min(__inputs_remaining, std::size_t{__max_inputs_per_block});
    const std::size_t __num_blocks = __inputs_remaining / __block_size + (__inputs_remaining % __block_size != 0);

    const bool __use_slm_for_comm = !std::is_trivially_copyable_v<T> || !dev.is_gpu();

    std::printf("[standalone_mat] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u use_slm=%d max_ipi=%u\n",
                __work_group_size, __num_work_groups, __inputs_per_item, __num_blocks,
                __max_num_sub_groups_global + 2, (int)__use_slm_for_comm, (unsigned)__max_inputs_per_item);

    // Scratch storage
    scratch_storage<T> __scratch(__max_num_sub_groups_global);

    // Buffers
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _InitType init{};

    // Instantiate submitters
    reduce_submitter<__max_inputs_per_item, __inclusive, _GenInput, _BinaryOp, _InitType>
        __reduce_sub{__num_work_groups, __work_group_size, __max_inputs_per_block,
                     __max_num_sub_groups_local, N, __use_slm_for_comm,
                     gen_input, binary_op, init};

    scan_submitter<__max_inputs_per_item, __inclusive, _BinaryOp, _GenInput, _ScanInputTransform, _WriteOp, _InitType>
        __scan_sub{__num_work_groups, __work_group_size, __max_inputs_per_block,
                   __max_num_sub_groups_local, __max_num_sub_groups_global,
                   __num_blocks, N, __use_slm_for_comm,
                   binary_op, gen_input, scan_xform, write_op, init};

    sycl::event __prior_event;
    std::size_t __remaining = N;

    buf_ref<T> __in_ref{buf_in, N};
    buf_ref<T> __out_ref{buf_out, N};

    for (std::size_t __b = 0; __b < __num_blocks; ++__b) {
        std::uint32_t __workitems_in_block = __dpl_ceiling_div(
            std::min(__remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_round_up =
            __dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __kernel_nd_range = sycl::nd_range<1>(sycl::range<1>(__workitems_round_up),
                                                    sycl::range<1>(__work_group_size));

        __prior_event = __reduce_sub(q, __kernel_nd_range, __in_ref, __scratch,
                                     __prior_event, __remaining, __b);
        __prior_event = __scan_sub(q, __kernel_nd_range, __in_ref, __out_ref, __scratch,
                                   __prior_event, __remaining, __b);

        __remaining -= std::min(__remaining, __block_size);
        if (__b + 2 == __num_blocks) {
            __inputs_per_item = __remaining >= __max_inputs_per_block
                ? __max_inputs_per_item
                : __dpl_ceiling_div(__dpl_bit_ceil(__remaining),
                                    __num_work_groups * __work_group_size);
        }
    }
    q.wait();

    // Read output
    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[standalone_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[standalone_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

    return errors ? 1 : 0;
}
