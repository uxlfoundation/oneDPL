// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan_tools.h --------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_TOOLS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_TOOLS_H

#include <cstddef> // std::size_t
#include <cstdint> // std::uint16_t
#include <utility> // std::forward

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// Index type within the temporary data array used to store per-diagonal results in registers.
using __temp_data_array_idx_t = std::uint16_t;

namespace __internal
{
// Temporary data stand-in which discards the stored values and instead captures
// the source position of the element at a specific index during a reduce then scan operation.
template <typename _FinalPosT>
struct __src_pos_capturing_temp_data
{
  public:
    // We should capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = true;

    __src_pos_capturing_temp_data(__temp_data_array_idx_t __idx_for_src_pos) : __idx_for_src_pos(__idx_for_src_pos) {}

    template <typename _ValueT2>
    void
    set(__temp_data_array_idx_t __idx, const _ValueT2&, _FinalPosT __src_idx)
    {
        if (__idx == __idx_for_src_pos)
            __saved_src_pos = __src_idx;
    }

    _FinalPosT
    __get_saved_src_pos() const
    {
        return __saved_src_pos;
    }

  private:
    const __temp_data_array_idx_t __idx_for_src_pos = 0;
    _FinalPosT __saved_src_pos = {};
};

// Tag type to indicate that no callback is provided
struct __no_callback_tag
{
};

// The second part of two-pass OOB processing: if the OOB position is reached in the first pass,
// here we recover the source indexes for the diagonal where it happened and store the OOB position from them.
template <typename _FinalPosT, typename _InRng, typename _OOBPositionT, typename _GenScanInput>
_FinalPosT
__finalize_oob_pos(_InRng&& __in_rng, _OOBPositionT __detected_oob_pos, const std::size_t __start_id_reached_on_oob,
                   _GenScanInput __gen_scan_input)
{
    __src_pos_capturing_temp_data<_FinalPosT> __pos_catcher(__detected_oob_pos);
    __gen_scan_input(std::forward<_InRng>(__in_rng), __start_id_reached_on_oob, __pos_catcher, __no_callback_tag{});
    return __pos_catcher.__get_saved_src_pos();
}

} // namespace __internal

// Temporary data structure which is used to store results to registers during a reduce then scan operation.
template <__temp_data_array_idx_t elements, typename _ValueT>
struct __temp_data_array
{
    // The maximum number of output elements a single scanned element may emit through this temporary data.
    // For set operations a scanned element is a diagonal which can produce up to `elements` outputs, so the
    // bounded-write estimate must account for this many writes per scanned element.
    static constexpr __temp_data_array_idx_t __max_outputs_per_input = elements;

    // We don't capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = false;

    template <typename _ValueT2>
    void
    set(__temp_data_array_idx_t __idx, const _ValueT2& __ele)
    {
        __data[__idx].__setup(__ele);
    }

    _ValueT
    get_and_destroy(__temp_data_array_idx_t __idx)
    {
        // Setting up temporary value to be destroyed as this function exits. The __scoped_destroyer calls destroy when
        // it leaves scope.
        oneapi::dpl::__internal::__scoped_destroyer<_ValueT> __destroy_when_leaving_scope{__data[__idx]};
        return __data[__idx].__v;
    }

    oneapi::dpl::__internal::__lazy_ctor_storage<_ValueT> __data[elements];
};

// This is a stand-in for a temporary data structure which is used to turn set() into a no-op. This is used in the case
// where no temporary register data is needed within reduce then scan kern
struct __noop_temp_data
{
    static constexpr __temp_data_array_idx_t __max_outputs_per_input = 1;

    // We don't capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = false;

    template <typename _ValueT>
    void
    set(__temp_data_array_idx_t, const _ValueT&) const
    {
    }
};

// Detecting TempData type alias in the specified structure
template <typename, typename = void>
struct __temp_data_required
{
    static constexpr bool value = false;
    using type = __noop_temp_data;
};

template <typename _T>
struct __temp_data_required<_T, std::void_t<typename _T::TempData>>
{
    static constexpr bool value = true;
    using type = typename _T::TempData;
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_TOOLS_H
