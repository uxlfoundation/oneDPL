// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan_pos_tools.h ----------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H

#include <algorithm>   // std::min
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint16_t
#include <limits>      // std::numeric_limits
#include <tuple>       // std::get
#include <type_traits> // std::integral_constant
#include <utility>     // std::forward

#include "../../onedpl_config.h"
#include "parallel_backend_sycl_utils.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// Describe final and OOB positions in source ranges together for bounded set operations
// (where tracking of OOB position is needed to determine the effective final position
// in source ranges based on output range size).
template <typename _Range1, typename _Range2>
struct _SetOpFinalAndOOBPosTypeImpl
{
    using _Size1 = oneapi::dpl::__internal::__difference_t<_Range1>;
    using _Size2 = oneapi::dpl::__internal::__difference_t<_Range2>;
    using _PositionT = oneapi::dpl::__internal::tuple<_Size1, _Size2>;
    using __final_pos_t = _PositionT;

    _PositionT __final_pos = {}; // Describes final position after set operation in source ranges
    _PositionT __oob_pos = {};   // Describes OOB position during set operation in source ranges

    // Compute the stop (end) position of the set operation result in the source ranges.
    // The effective stop position is the smaller of:
    //   - the final position produced by the set operation (without output range size limitations), and
    //   - the out-of-bounds (OOB) position marking the end of the available output range.
    std::tuple<_Size1, _Size2>
    __compute_stop_pos() const
    {
        return {std::min(std::get<0>(__final_pos), std::get<0>(__oob_pos)),
                std::min(std::get<1>(__final_pos), std::get<1>(__oob_pos))};
    }

    template <typename _SetTag>
    static _SetOpFinalAndOOBPosTypeImpl
    __create_initial_state(_SetTag __set_tag, const _Range1& __range1, const _Range2& __range2)
    {
        const _Size1 __n1 = oneapi::dpl::__ranges::__size(__range1);
        const _Size2 __n2 = oneapi::dpl::__ranges::__size(__range2);

        _PositionT __initial_final_pos = {__n1, __n2}; // set_union and set_symmetric_difference stop at the ends
        if constexpr (std::is_same_v<_SetTag, unseq_backend::_DifferenceTag>)
            __initial_final_pos = {__n1, 0}; // for set_difference, position in the second range is obtained later
        else if constexpr (std::is_same_v<_SetTag, unseq_backend::_IntersectionTag>)
            __initial_final_pos = {0, 0}; // for set_intersection, both positions are obtained later

        // Initial OOB state initialized by the size of the ranges because we will use std::min() for their state
        // and final positions on host side after finish Kernel's code.
        _PositionT __initial_oob_pos{__n1, __n2};

        return _SetOpFinalAndOOBPosTypeImpl{__initial_final_pos, __initial_oob_pos};
    }
};

template <typename _Range1, typename _Range2>
using _SetOpFinalAndOOBPosType = _SetOpFinalAndOOBPosTypeImpl<std::decay_t<_Range1>, std::decay_t<_Range2>>;

namespace __internal
{
// Describes whether we have a final-position type in the storage or not
template <typename _StopPosStorage>
inline constexpr bool __has_final_pos = false;

template <typename _Range1, typename _Range2>
inline constexpr bool __has_final_pos<_SetOpFinalAndOOBPosTypeImpl<_Range1, _Range2>> = true;

// Temporary data stand-in which discards the stored values and instead captures
// the source position of the element at a specific index during a reduce then scan operation.
template <typename _FinalPosT>
struct __src_pos_capturing_temp_data
{
  public:
    // We should capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = true;

    __src_pos_capturing_temp_data(std::uint16_t __idx_for_src_pos) : __idx_for_src_pos(__idx_for_src_pos) {}

    template <typename _ValueT2>
    void
    set(std::uint16_t __idx, const _ValueT2&, _FinalPosT __src_idx)
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
    const std::uint16_t __idx_for_src_pos = 0;
    _FinalPosT __saved_src_pos = {};
};

// Tag type to indicate that no callback is provided
struct __no_callback_tag
{
};

// Sentinel type used as a stand-in for the stop-position accessor when _Bounded=false.
struct __no_stop_pos_acc_tag
{
    // No stop position is tracked when _Bounded=false, so std::size_t is just a harmless default.
    using type = std::size_t;
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

template <bool _Bounded, typename _StopPosInitState>
auto
__create_stop_pos_storage_opt(sycl::queue& __q)
{
    if constexpr (_Bounded)
        return __result_storage<_StopPosInitState>(__q, 1);
    else
        return __internal::__no_stop_pos_acc_tag{};
}

template <bool _Bounded, typename _ModeTagT, typename _StopPosStorage>
auto
__get_stop_pos_accessor_opt(_ModeTagT __mode, sycl::handler& __cgh, _StopPosStorage& __stop_pos_storage)
{
    if constexpr (_Bounded)
        return __get_accessor(__mode, __stop_pos_storage, __cgh, __dpl_sycl::__no_init{});
    else
        return __no_stop_pos_acc_tag{};
}

} // namespace __internal

// The return type of set operation implementation for bounded and unbounded cases.
// For bounded case we need to return final and OOB positions in source ranges,
// for unbounded case we need to return only the size of the result.
template <bool _Bounded, typename _Range1, typename _Range2, typename _Range3>
using __set_op_impl_return_t =
    std::conditional_t<_Bounded,
                       oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::__difference_t<_Range1>,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>,
                       oneapi::dpl::__internal::__difference_t<_Range3>>;

// Create the result of set operation implementation for bounded and unbounded cases
// based on the final and OOB positions in source ranges or the size of the result respectively.
template <bool _Bounded, typename _Range1, typename _Range2, typename _Range3>
__set_op_impl_return_t<_Bounded, _Range1, _Range2, _Range3>
__create_set_op_impl_result(oneapi::dpl::__internal::__difference_t<_Range1> __idx1,
                            oneapi::dpl::__internal::__difference_t<_Range2> __idx2,
                            oneapi::dpl::__internal::__difference_t<_Range3> __idx3)
{
    if constexpr (_Bounded)
        return {__idx1, __idx2, __idx3};
    else
        return __idx3;
}

template <bool _Bounded, typename _SetTag, typename _Range1, typename _Range2>
std::conditional_t<_Bounded, _SetOpFinalAndOOBPosType<_Range1, _Range2>, std::size_t>
__create_initial_final_and_oob_pos_state(_SetTag __set_tag, const _Range1& __range1, const _Range2& __range2)
{
    if constexpr (_Bounded)
        return _SetOpFinalAndOOBPosType<_Range1, _Range2>::__create_initial_state(__set_tag, __range1, __range2);
    else
        return std::size_t{0};
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H
