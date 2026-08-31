// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_RANGE_PACK_H
#define _ONEDPL_KT_RANGE_PACK_H

#include <type_traits>
#include <utility>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <typename _Rng>
auto
__rng_data(const _Rng& __rng)
{
    return __rng.begin();
}

// ESIMD requires the accessor directly: sycl::accessor::operator[] needs -fsycl-esimd-force-stateless-mem.
// TODO: rely on begin() once -fsycl-esimd-force-stateless-mem has been enabled by default
template <typename _T, sycl::access::mode _M, bool _NoInit>
auto
__rng_data(const oneapi::dpl::__ranges::all_view<_T, _M, _NoInit>& __view)
{
    return __view.accessor();
}

struct __range_dummy
{
};

template <typename _Rng>
struct __range_value_type_deducer
{
    using __value_t = oneapi::dpl::__internal::__value_t<_Rng>;
};

template <>
struct __range_value_type_deducer<__range_dummy>
{
    using __value_t = void;
};

template <typename _Rng1, typename _Rng2 = __range_dummy>
struct __range_pack
{
    using _KeyT = typename __range_value_type_deducer<_Rng1>::__value_t;
    using _ValT = typename __range_value_type_deducer<_Rng2>::__value_t;
    static constexpr bool __has_values = !std::is_void_v<_ValT>;

    const auto&
    __keys_rng() const
    {
        return __m_keys_rng;
    }
    const auto&
    __vals_rng() const
    {
        static_assert(__has_values);
        return __m_vals_rng;
    }

    __range_pack(const _Rng1& __rng1, const _Rng2& __rng2 = __range_dummy{}) : __m_keys_rng(__rng1), __m_vals_rng(__rng2) {}
    __range_pack(_Rng1&& __rng1, _Rng2&& __rng2 = __range_dummy{})
        : __m_keys_rng(::std::move(__rng1)), __m_vals_rng(::std::move(__rng2))
    {
    }

  private:
    _Rng1 __m_keys_rng;
    _Rng2 __m_vals_rng;
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RANGE_PACK_H
