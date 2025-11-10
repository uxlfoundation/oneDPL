/*
 *  Copyright (c) Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_GCD_IMPL_H
#define _ONEDPL_GCD_IMPL_H

#include <limits>
#include <type_traits>

namespace oneapi
{
namespace dpl
{

namespace internal
{
template <typename _Result, typename _Source>
_Result
__abs_impl(_Source __t, ::std::true_type)
{
    if (__t >= 0)
        return __t;
    if (__t == ::std::numeric_limits<_Source>::min())
        return -static_cast<_Result>(__t);
    return -__t;
}

template <typename _Result, typename _Source>
_Result
__abs_impl(_Source __t, ::std::false_type)
{
    return __t;
}

template <typename _Result, typename _Source>
constexpr _Result
__get_abs(_Source __t)
{
    return __abs_impl<_Result>(__t, ::std::is_signed<_Source>{});
}

} // namespace internal

// Some C++ standard libraries implement std::gcd and std::lcm as recursive functions,
// which prevents their use in SYCL kernels (see the SYCL specification for more details).
// Therefore oneDPL provides its own implementation.

// gcd
template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
gcd(_Mn __m, _Nn __n)
{
    static_assert((::std::is_integral_v<_Mn> && ::std::is_integral_v<_Nn>), "Arguments to gcd must be integer types");
    static_assert((!::std::is_same_v<::std::remove_cv_t<_Mn>, bool>), "First argument to gcd cannot be bool");
    static_assert((!::std::is_same_v<::std::remove_cv_t<_Nn>, bool>), "Second argument to gcd cannot be bool");
    using _Rp = ::std::common_type_t<_Mn, _Nn>;
    using _Wp = ::std::make_unsigned_t<_Rp>;
    _Wp __m1 = static_cast<_Wp>(oneapi::dpl::internal::__get_abs<_Rp>(__m));
    _Wp __n1 = static_cast<_Wp>(oneapi::dpl::internal::__get_abs<_Rp>(__n));

    while (__n1 != 0)
    {
        _Wp __t = __m1 % __n1;
        __m1 = __n1;
        __n1 = __t;
    }
    return static_cast<_Rp>(__m1);
}

// lcm
template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
lcm(_Mn __m, _Nn __n)
{
    static_assert((::std::is_integral_v<_Mn> && ::std::is_integral_v<_Nn>), "Arguments to lcm must be integer types");
    static_assert((!::std::is_same_v<::std::remove_cv_t<_Mn>, bool>), "First argument to lcm cannot be bool");
    static_assert((!::std::is_same_v<::std::remove_cv_t<_Nn>, bool>), "Second argument to lcm cannot be bool");
    if (__m == 0 || __n == 0)
        return 0;
    using _Rp = ::std::common_type_t<_Mn, _Nn>;
    _Rp __val1 = oneapi::dpl::internal::__get_abs<_Rp>(__m) / gcd(__m, __n);
    _Rp __val2 = oneapi::dpl::internal::__get_abs<_Rp>(__n);
    return __val1 * __val2;
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_GCD_IMPL_H
