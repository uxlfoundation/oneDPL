// -*- C++ -*-
//===-- philox_engine.h ---------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Public header file provides implementation for Philox Engine

#ifndef _ONEDPL_PHILOX_ENGINE_H
#define _ONEDPL_PHILOX_ENGINE_H

#include <cstdint>
#include <utility>
#include <type_traits>
#include <limits>
#include <array>
#include <istream>
#include <ostream>
#include <algorithm>

#include "random_common.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

template <typename _UIntType, std::size_t _w, std::size_t _n, std::size_t _r,
          internal::element_type_t<_UIntType>... _consts>
class philox_engine;

template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
std::basic_ostream<__CharT, __Traits>&
operator<<(std::basic_ostream<__CharT, __Traits>&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
const sycl::stream&
operator<<(const sycl::stream&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
std::basic_istream<__CharT, __Traits>&
operator>>(std::basic_istream<__CharT, __Traits>&, philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <typename _UIntType, std::size_t _w, std::size_t _n, std::size_t _r,
          internal::element_type_t<_UIntType>... _consts>
class philox_engine
{
  public:
    /* Types */
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

  private:
    /* The size of the consts arrays */
    static constexpr std::size_t __array_size = _n / 2;

    /* Methods for unpacking variadic of constants into two arrays */
    template <std::size_t... _Is>
    static constexpr auto
    get_even_element_array(std::array<scalar_type, _n> __input_array, std::index_sequence<_Is...>)
    {
        return std::array<scalar_type, sizeof...(_Is)>{__input_array[_Is * 2]...};
    }
    template <std::size_t... _Is>
    static constexpr auto
    get_odd_element_array(std::array<scalar_type, _n> __input_array, std::index_sequence<_Is...>)
    {
        return std::array<scalar_type, sizeof...(_Is)>{__input_array[_Is * 2 + 1]...};
    }

  public:
    /* Engine characteristics */
    static constexpr std::size_t word_size = _w;
    static constexpr std::size_t word_count = _n;
    static constexpr std::size_t round_count = _r;

    static_assert(_n == 2 || _n == 4, "parameter n must be 2 or 4");
    static_assert(sizeof...(_consts) == _n, "the amount of consts must be equal to n");
    static_assert(_r > 0, "parameter r must be more than 0");
    static_assert(_w > 0 && _w <= std::numeric_limits<scalar_type>::digits,
                  "parameter w must satisfy 0 < w < std::numeric_limits<UIntType>::digits");
    static_assert(std::numeric_limits<scalar_type>::digits <= 64,
                  "size of the scalar UIntType (in case of sycl::vec<T, N> the size of T) must be less than 64 bits");
    static_assert(std::is_unsigned_v<scalar_type>, "UIntType must be unsigned type or vector of unsigned types");

    static constexpr std::array<scalar_type, __array_size> multipliers =
        get_even_element_array(std::array{_consts...}, std::make_index_sequence<__array_size>{});
    static constexpr std::array<scalar_type, __array_size> round_consts =
        get_odd_element_array(std::array{_consts...}, std::make_index_sequence<__array_size>{});

    static constexpr scalar_type
    min()
    {
        return 0;
    }

    static constexpr scalar_type
    max()
    {
        // equals to 2^w - 1
        return in_mask;
    }

    static constexpr scalar_type default_seed = 20111115u;

    /* Constructors */
    philox_engine() : philox_engine(default_seed) {}

    explicit philox_engine(scalar_type __seed) { seed(__seed); }

    /* Seeding function */
    void
    seed(scalar_type __seed = default_seed)
    {
        seed_internal(__seed & in_mask);
    }

    /* Set the state to arbitrary position */
    void
    set_counter(const std::array<scalar_type, word_count>& __counter)
    {
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            // all counters are set in reverse order
            state_.X[word_count - __i - 1] = __counter[__i] & in_mask;
        }
    }

    /* Generating functions */
    result_type
    operator()()
    {
        return generate_internal<internal::type_traits_t<result_type>::num_elems>();
    }

    /* operator () overload for result portion generation */
    result_type
    operator()(unsigned int __random_nums)
    {
        return generate_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
    }

    /* Shift the counter only forward relative to its current position */
    void
    discard(unsigned long long __z)
    {
        discard_internal(__z);
    }

    /* Equality operators */
    friend bool
    operator==(const philox_engine& __x, const philox_engine& __y)
    {
        return (std::equal(__x.state_.X.begin(), __x.state_.X.end(), __y.state_.X.begin()) &&
                std::equal(__x.state_.K.begin(), __x.state_.K.end(), __y.state_.K.begin()) &&
                std::equal(__x.state_.Y.begin(), __x.state_.Y.end(), __y.state_.Y.begin()) &&
                __x.state_.idx == __y.state_.idx);
    }

    friend bool
    operator!=(const philox_engine& __x, const philox_engine& __y)
    {
        return !(__x == __y);
    }

    /* Inserters and extractors */
    template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n,
              std::size_t __r, internal::element_type_t<__UIntType>... __consts>
    friend std::basic_ostream<__CharT, __Traits>&
    operator<<(std::basic_ostream<__CharT, __Traits>&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

    template <typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
              internal::element_type_t<__UIntType>... __consts>
    friend const sycl::stream&
    operator<<(const sycl::stream&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

    template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n,
              std::size_t __r, internal::element_type_t<__UIntType>... __consts>
    friend std::basic_istream<__CharT, __Traits>&
    operator>>(std::basic_istream<__CharT, __Traits>&, philox_engine<__UIntType, __w, __n, __r, __consts...>&);

  private:
    /* Internal generator state */
    struct state
    {
        std::array<scalar_type, word_count> X;     // counters
        std::array<scalar_type, word_count / 2> K; // keys
        std::array<scalar_type, word_count> Y;     // results
        scalar_type idx;                           // index
    } state_;

    /* __word_mask<_W> - scalar_type with the low _W bits set */
    template <std::size_t _W, typename = std::enable_if_t<_W != 0>>
    static constexpr scalar_type __word_mask = ~scalar_type(0) >> (std::numeric_limits<scalar_type>::digits - _W);

    /* Processing mask */
    static constexpr auto in_mask = __word_mask<word_size>;

    void
    seed_internal(scalar_type __seed)
    {
        // set to zero counters and results
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            state_.X[__i] = 0;
            state_.Y[__i] = 0;
        }
        // 0th key element is set as seed, others are 0
        state_.K[0] = __seed & in_mask;
        for (std::size_t __i = 1; __i < (word_count / 2); ++__i)
        {
            state_.K[__i] = 0;
        }

        state_.idx = word_count - 1;
    }

    /* Increment counter by 1 */
    void
    increase_counter_internal()
    {
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            state_.X[__i] = (state_.X[__i] + 1) & in_mask;
            if (state_.X[__i])
            {
                return;
            }
        }
    }

    /* Increment counter by an arbitrary __z */
    void
    increase_counter_internal(unsigned long long __z)
    {
        unsigned long long __carry = 0;
        unsigned long long __ctr_inc = __z;

        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            scalar_type __initial_x = state_.X[__i];
            state_.X[__i] = (__initial_x + __ctr_inc + __carry) & in_mask;

            __carry = 0;
            // check overflow of the current chunk
            if (state_.X[__i] < __initial_x)
            {
                __carry = 1;
            }

            //          select high chunk             shift for addition with the next counter chunk
            __ctr_inc = (__ctr_inc & (~in_mask)) >> (std::numeric_limits<unsigned long long>::digits - word_size);
        }
    }

    /* generate_internal() specified for sycl_vec output 
       and overload for result portion generation */
    template <unsigned int _N>
    std::enable_if_t<(_N > 0), result_type>
    generate_internal(unsigned int __random_nums)
    {
        if (__random_nums >= _N)
            return operator()();

        result_type __loc_result;
        for (int __elm_count = 0; __elm_count < __random_nums; ++__elm_count)
        {
            ++state_.idx;

            // check if buffer is empty
            if (state_.idx == word_count)
            {
                philox_kernel();
                increase_counter_internal();
                state_.idx = 0;
            }
            __loc_result[__elm_count] = state_.Y[state_.idx];
        }

        return __loc_result;
    }

    /* generate_internal() specified for sycl_vec output */
    template <unsigned int _N>
    std::enable_if_t<(_N > 0), result_type>
    generate_internal()
    {
        result_type __loc_result;
        for (int __elm_count = 0; __elm_count < _N; ++__elm_count)
        {
            ++state_.idx;

            // check if buffer is empty
            if (state_.idx == word_count)
            {
                philox_kernel();
                increase_counter_internal();
                state_.idx = 0;
            }
            __loc_result[__elm_count] = state_.Y[state_.idx];
        }

        return __loc_result;
    }

    /* generate_internal() specified for a scalar output */
    template <unsigned int _N>
    std::enable_if_t<(_N == 0), result_type>
    generate_internal()
    {
        ++state_.idx;
        if (state_.idx == word_count)
        {
            philox_kernel();
            increase_counter_internal();
            state_.idx = 0;
        }

        return state_.Y[state_.idx];
    }

    void
    discard_internal(unsigned long long __z)
    {
        std::uint32_t __available_in_buffer = word_count - 1 - state_.idx;
        if (__z <= __available_in_buffer)
        {
            state_.idx += __z;
        }
        else
        {
            __z -= __available_in_buffer;
            int __tail = __z % word_count;
            if (__tail == 0)
            {
                increase_counter_internal(__z / word_count);
                state_.idx = word_count - 1;
            }
            else
            {
                if (__z > word_count)
                {
                    increase_counter_internal((__z - 1) / word_count);
                }
                philox_kernel();
                increase_counter_internal();
                state_.idx = __tail - 1;
            }
        }
    }

    /* Internal generation Philox kernel */
    void
    philox_kernel()
    {
        if constexpr (word_count == 2)
        {
            scalar_type __V0 = state_.X[0];
            scalar_type __V1 = state_.X[1];
            scalar_type __K0 = state_.K[0];
            for (std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = mulhilo(__V0, multipliers[0]);
                __V0 = __hi0 ^ __K0 ^ __V1;
                __V1 = __lo0;
                __K0 = (__K0 + round_consts[0]) & in_mask;
            }
            state_.Y[0] = __V0;
            state_.Y[1] = __V1;
        }
        else if constexpr (word_count == 4)
        {
            // permute X to V
            scalar_type __V2 = state_.X[0];
            scalar_type __V1 = state_.X[1];
            scalar_type __V0 = state_.X[2];
            scalar_type __V3 = state_.X[3];
            scalar_type __K0 = state_.K[0];
            scalar_type __K1 = state_.K[1];
            for (std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = mulhilo(__V0, multipliers[0]);
                auto [__hi1, __lo1] = mulhilo(__V2, multipliers[1]);
                __V2 = __hi0 ^ __V1 ^ __K0;
                __V1 = __lo0;
                __V0 = __hi1 ^ __V3 ^ __K1;
                __V3 = __lo1;
                __K0 = (__K0 + round_consts[0]) & in_mask;
                __K1 = (__K1 + round_consts[1]) & in_mask;
            }
            state_.Y[0] = __V2;
            state_.Y[1] = __V1;
            state_.Y[2] = __V0;
            state_.Y[3] = __V3;
        }
    }

    /* Returns the word_size high and word_size low
       bits of the 2*word_size-bit product of __a and __b */
    static std::pair<scalar_type, scalar_type>
    mulhilo(scalar_type __a, scalar_type __b)
    {
        scalar_type __res_hi, __res_lo;

        /* multiplication fits standard types */
        if constexpr (word_size <= 32)
        {
            std::uint_fast64_t __mult_result = (std::uint_fast64_t)__a * (std::uint_fast64_t)__b;
            __res_hi = (__mult_result >> word_size) & in_mask;
            __res_lo = __mult_result & in_mask;
        }
        /* pen-pencil multiplication by 32-bit chunks */
        else if constexpr (word_size > 32)
        {
            constexpr std::size_t chunk_size = 32;
            __res_lo = (__a * __b) & in_mask;

            scalar_type __x0 = __a & __word_mask<chunk_size>;
            scalar_type __x1 = __a >> chunk_size;
            scalar_type __y0 = __b & __word_mask<chunk_size>;
            scalar_type __y1 = __b >> chunk_size;

            scalar_type __p11 = __x1 * __y1;
            scalar_type __p01 = __x0 * __y1;
            scalar_type __p10 = __x1 * __y0;
            scalar_type __p00 = __x0 * __y0;

            // 64-bit product + two 32-bit values
            scalar_type __middle = __p10 + (__p00 >> chunk_size) + (__p01 & __word_mask<chunk_size>);

            // 64-bit product + two 32-bit values
            __res_hi = (__p11 + (__middle >> chunk_size) + (__p01 >> chunk_size)) & in_mask;
        }

        return {__res_hi, __res_lo};
    }
};

template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
std::basic_ostream<__CharT, __Traits>&
operator<<(std::basic_ostream<__CharT, __Traits>& __os,
           const philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    internal::save_stream_flags<__CharT, __Traits> __flags(__os);

    __os.setf(std::ios_base::dec | std::ios_base::left);
    __CharT __sp = __os.widen(' ');
    __os.fill(__sp);

    for (auto x_elm : __engine.state_.X)
    {
        __os << x_elm << __sp;
    }
    for (auto k_elm : __engine.state_.K)
    {
        __os << k_elm << __sp;
    }
    for (auto y_elm : __engine.state_.Y)
    {
        __os << y_elm << __sp;
    }
    __os << __engine.state_.idx;

    return __os;
}

template <typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
const sycl::stream&
operator<<(const sycl::stream& __os, const philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    for (auto __x_elm : __engine.state_.X)
    {
        __os << __x_elm << ' ';
    }
    for (auto __k_elm : __engine.state_.K)
    {
        __os << __k_elm << ' ';
    }
    for (auto __y_elm : __engine.state_.Y)
    {
        __os << __y_elm << ' ';
    }
    __os << __engine.state_.idx;

    return __os;
}

template <typename __CharT, typename __Traits, typename __UIntType, std::size_t __w, std::size_t __n, std::size_t __r,
          internal::element_type_t<__UIntType>... __consts>
std::basic_istream<__CharT, __Traits>&
operator>>(std::basic_istream<__CharT, __Traits>& __is, philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    internal::save_stream_flags<__CharT, __Traits> __flags(__is);

    __is.setf(std::ios_base::dec);

    const std::size_t __state_size = 2 * __n + __n / 2 + 1;

    std::array<internal::element_type_t<__UIntType>, __state_size> __tmp_inp;
    for (std::size_t __i = 0; __i < __state_size; ++__i)
    {
        __is >> __tmp_inp[__i];
    }

    if (!__is.fail())
    {
        int __inp_itr = 0;
        for (std::size_t __i = 0; __i < __n; ++__i, ++__inp_itr)
            __engine.state_.X[__i] = __tmp_inp[__inp_itr];
        for (std::size_t __i = 0; __i < __n / 2; ++__i, ++__inp_itr)
            __engine.state_.K[__i] = __tmp_inp[__inp_itr];
        for (std::size_t __i = 0; __i < __n; ++__i, ++__inp_itr)
            __engine.state_.Y[__i] = __tmp_inp[__inp_itr];
        __engine.state_.idx = __tmp_inp[__inp_itr];
    }

    return __is;
}

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H
