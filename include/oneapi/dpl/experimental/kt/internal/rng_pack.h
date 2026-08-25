// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_RNG_PACK_H
#define _ONEDPL_KT_RNG_PACK_H

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

// ESIMD functionality requires using an accessor directly due to the restriction:
//      sycl::accessor::operator[] are supported only with -fsycl-esimd-force-stateless-mem.
//      Otherwise, all memory accesses through an accessor are done via explicit APIs
// TODO: rely on begin() once -fsycl-esimd-force-stateless-mem has been enabled by default
template <typename _T, sycl::access::mode _M, bool _NoInit>
auto
__rng_data(const oneapi::dpl::__ranges::all_view<_T, _M, _NoInit>& __view)
{
    return __view.accessor();
}

struct __rng_dummy
{
};

template <typename _Rng>
struct __rng_value_type_deducer
{
    using __value_t = oneapi::dpl::__internal::__value_t<_Rng>;
};

template <>
struct __rng_value_type_deducer<__rng_dummy>
{
    using __value_t = void;
};

template <typename _Rng1, typename _Rng2 = __rng_dummy>
struct __rng_pack
{
    using _KeyT = typename __rng_value_type_deducer<_Rng1>::__value_t;
    using _ValT = typename __rng_value_type_deducer<_Rng2>::__value_t;
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

    __rng_pack(const _Rng1& __rng1, const _Rng2& __rng2 = __rng_dummy{}) : __m_keys_rng(__rng1), __m_vals_rng(__rng2) {}
    __rng_pack(_Rng1&& __rng1, _Rng2&& __rng2 = __rng_dummy{})
        : __m_keys_rng(::std::move(__rng1)), __m_vals_rng(::std::move(__rng2))
    {
    }

  private:
    _Rng1 __m_keys_rng;
    _Rng2 __m_vals_rng;
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RNG_PACK_H
