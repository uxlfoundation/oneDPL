// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan_pos_tools.h ----------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint16_t
#include <limits>      // std::numeric_limits
#include <tuple>       // std::get
#include <type_traits> // std::integral_constant

#include "parallel_backend_sycl_utils.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
namespace __internal
{

template <typename, typename = void>
struct __select_max_outputs_per_input : std::integral_constant<std::uint16_t, 1>
{
    // By default, each work-item emits at most one output per scanned element.
};

template <typename _T>
struct __select_max_outputs_per_input<_T, std::void_t<decltype(_T::__max_outputs_per_input)>>
    : std::integral_constant<std::uint16_t, _T::__max_outputs_per_input>
{
};

template <typename _T>
constexpr std::uint16_t __select_max_outputs_per_input_v = __select_max_outputs_per_input<std::decay_t<_T>>::value;

// Temporary data stand-in which discards the stored values and instead captures
// the source position of the element at a specific index during a reduce then scan operation.
template <typename _SrcDataPosT>
struct __src_pos_capturing_temp_data
{
  public:
    __src_pos_capturing_temp_data(std::uint16_t __idx_for_src_pos) : __idx_for_src_pos(__idx_for_src_pos) {}

    template <typename _ValueT2>
    void
    set(std::uint16_t __idx, const _ValueT2&, _SrcDataPosT __src_idx)
    {
        if (__idx == __idx_for_src_pos)
            __saved_src_pos = __src_idx;
    }

    _SrcDataPosT
    __get_saved_src_pos() const
    {
        return __saved_src_pos;
    }

  private:
    const std::uint16_t __idx_for_src_pos = 0;
    _SrcDataPosT __saved_src_pos = {};
};

template <typename = void>
struct __temp_data_capture_indexes_flag : std::false_type
{
};

template <typename _SrcDataPosT>
struct __temp_data_capture_indexes_flag<__src_pos_capturing_temp_data<_SrcDataPosT>> : std::true_type
{
};

template <typename _TempData>
inline constexpr bool __temp_data_capture_indexes_flag_v =
    __temp_data_capture_indexes_flag<std::decay_t<_TempData>>::value;

// Tag type to indicate that no callback is provided
struct __no_callback_tag
{
};

// Helper variable template to check if a callback is provided or not
template <typename _TCallback>
inline constexpr bool __is_no_callback_v = std::is_same_v<std::decay_t<_TCallback>, __no_callback_tag>;

template <typename, typename = void>
struct __detect_oob_in_two_steps_selector : std::false_type
{
};

template <typename _T>
inline constexpr bool __detect_oob_in_two_steps_v = __detect_oob_in_two_steps_selector<std::decay_t<_T>>::value;

// Sentinel type used as a stand-in for the stop-position accessor when _Bounded=false.
struct __no_stop_pos_acc_tag
{
};

template <typename _T>
inline constexpr bool __is_no_stop_pos_acc_v = std::is_same_v<std::remove_cv_t<_T>, __no_stop_pos_acc_tag>;

template <bool _Bounded, typename _Mode, typename _StopPosStorage>
auto
__get_stop_pos_accessor_opt(_Mode __mode, sycl::handler& __cgh, _StopPosStorage& __stop_pos_storage)
{
    if constexpr (_Bounded)
    {
        return __get_accessor(__mode, __stop_pos_storage, __cgh, __dpl_sycl::__no_init{});
    }
    else
    {
        return __no_stop_pos_acc_tag{};
    }
}

template <typename _StopPosStorage, typename = void>
struct __stop_pos_storage_type_selector
{
    using type = std::size_t;
};

template <typename _StopPosStorage>
struct __stop_pos_storage_type_selector<_StopPosStorage, std::void_t<typename _StopPosStorage::type>>
{
    using type = typename _StopPosStorage::type;
};

template <typename _StopPosStorage>
using __stop_pos_storage_type_selector_t = typename __stop_pos_storage_type_selector<_StopPosStorage>::type;

template <bool _Bounded, typename _GenScanInput, typename _StopPosStorage>
struct __parallel_reduce_then_scan_stop_oob_pos_tools
{
    using __storage_data_t = __stop_pos_storage_type_selector_t<_StopPosStorage>;

    // Describes whether we have a final-position type in the storage or not
    static constexpr bool __has_src_final_pos =
        oneapi::dpl::__ranges::__internal::__has_final_pos_type_v<__storage_data_t>;

    // Describes final position type (if any) in the storage
    using __src_final_pos_t = oneapi::dpl::__ranges::__internal::__final_pos_type_selector_t<__storage_data_t>;

    // Describes OOB position type
    using __oob_pos_t =
        std::conditional_t<__detect_oob_in_two_steps_v<_GenScanInput>, std::uint16_t, __src_final_pos_t>;

    static __oob_pos_t
    __create_initial_oob_pos()
    {
        if constexpr (std::is_arithmetic_v<__oob_pos_t>)
            return std::numeric_limits<__oob_pos_t>::max();
        else
            return {};
    }

    static bool
    __is_eq_to_initial_oob_pos(const __oob_pos_t& __pos)
    {
        return __pos == __create_initial_oob_pos();
    }

    template <typename _Group, typename _FinalPosType>
    static _FinalPosType
    __reduce_max_final_pos(_Group __group, _FinalPosType& __last_idxs_in_this_wi)
    {
        if constexpr (std::is_arithmetic_v<std::decay_t<_FinalPosType>>)
        {
            return __dpl_sycl::__reduce_over_group(__group, __last_idxs_in_this_wi,
                                                   __dpl_sycl::__maximum<_FinalPosType>());
        }
        else
        {
            using TField0 = std::decay_t<decltype(std::get<0>(__last_idxs_in_this_wi))>;
            using TField1 = std::decay_t<decltype(std::get<1>(__last_idxs_in_this_wi))>;

            return {__dpl_sycl::__reduce_over_group(__group, std::get<0>(__last_idxs_in_this_wi),
                                                    __dpl_sycl::__maximum<TField0>()),
                    __dpl_sycl::__reduce_over_group(__group, std::get<1>(__last_idxs_in_this_wi),
                                                    __dpl_sycl::__maximum<TField1>())};
        }
    }

    template <typename __FinalAndOOBPosAcc, typename _OOBPosType>
    static void
    __update_oob_pos(__FinalAndOOBPosAcc& __final_and_oob_pos_acc, const _OOBPosType& __oob_pos)
    {
        auto& __final_and_oob_pos = __final_and_oob_pos_acc.__data()[0];

        // No synchronization needed because OOB position may be reached only in a single work-item
        if constexpr (std::is_arithmetic_v<std::decay_t<_OOBPosType>>)
        {
            // The __final_and_oob_pos really is simple position value
            __final_and_oob_pos = __oob_pos;
        }
        else
        {
            // The __final_and_oob_pos really is _SetOpFinalAndOOBPosType
            __final_and_oob_pos.__oob_pos = __oob_pos;
        }
    }

    // Device-scope atomic over global memory. The target must live in device USM
    // (see __transform_reduce_then_scan_stop_pos_storage_t); a device-scope atomic on host USM
    // triggers an atomic access violation on device.
    template <typename _FieldT>
    using _StopPosFieldAtomicRefT =
        sycl::atomic_ref<std::decay_t<_FieldT>, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>;

    template <typename __FinalAndOOBPosAcc, typename _FinalPosType>
    static void
    __update_final_pos(__FinalAndOOBPosAcc& __final_and_oob_pos_acc, const _FinalPosType& __final_pos)
    {
        // The __final_and_oob_pos really is _SetOpFinalAndOOBPosType
        auto& __final_and_oob_pos = __final_and_oob_pos_acc.__data()[0];

        // We need atomic access here as multiple work-items can setup reached final position and update it concurrently.
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(__final_and_oob_pos)>>)
        {
            _StopPosFieldAtomicRefT<_FinalPosType>(__final_and_oob_pos).fetch_max(__final_pos);
        }
        else
        {
            auto& __final_pos_field0 = std::get<0>(__final_and_oob_pos.__final_pos);
            auto& __final_pos_field1 = std::get<1>(__final_and_oob_pos.__final_pos);

            using _FinalPosType0 = std::decay_t<decltype(__final_pos_field0)>;
            using _FinalPosType1 = std::decay_t<decltype(__final_pos_field1)>;

            _StopPosFieldAtomicRefT<_FinalPosType0>(__final_pos_field0).fetch_max(std::get<0>(__final_pos));
            _StopPosFieldAtomicRefT<_FinalPosType1>(__final_pos_field1).fetch_max(std::get<1>(__final_pos));
        }
    }

    template <typename __oob_pos_t>
    static auto
    __create_on_oob_reached(std::size_t& __start_id_reached, std::size_t& __start_id_reached_on_oob,
                            __oob_pos_t& __oob_detected)
    {
        return [&__start_id_reached, &__start_id_reached_on_oob, &__oob_detected](__oob_pos_t __id) {
            __start_id_reached_on_oob = __start_id_reached;
            __oob_detected = __id;
        };
    }

    template <typename _FinalPosType>
    static auto
    __create_final_pos_saver(_FinalPosType& __src_final_pos)
    {
        if constexpr (__has_src_final_pos)
            return [&__src_final_pos](_FinalPosType __final_pos) { __src_final_pos = __final_pos; };
        else
            return __no_callback_tag{};
    }

    template <typename _OOBPositionT, typename _GenScanInputArg>
    static std::conditional_t<__detect_oob_in_two_steps_v<_GenScanInput>, __src_final_pos_t, _OOBPositionT>
    __finalize_oob_detected(_OOBPositionT __detected_oob_pos, const std::size_t __start_id_reached_on_oob,
                            _GenScanInputArg __gen_scan_input)
    {
        if constexpr (__detect_oob_in_two_steps_v<_GenScanInput>)
        {
            __src_pos_capturing_temp_data<__src_final_pos_t> __pos_catcher(__detected_oob_pos);
            __gen_scan_input(__start_id_reached_on_oob, __pos_catcher, __no_callback_tag{});
            return __pos_catcher.__get_saved_src_pos();
        }
        else
        {
            return __detected_oob_pos;
        }
    }

    static auto
    __create_src_final_pos_sg_container()
    {
        if constexpr (_Bounded && __has_src_final_pos)
            return __src_final_pos_t{};
        else
            return 0;
    }
};

} // namespace __internal
} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_POS_TOOLS_H
