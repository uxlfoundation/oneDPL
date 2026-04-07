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
//
// The documentation of the Engine: https://en.cppreference.com/w/cpp/numeric/random/philox_engine
//

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

template <typename _UIntType, std::size_t _W, std::size_t _N, std::size_t _R,
          oneapi::dpl::internal::element_type_t<_UIntType>... _Consts>
class philox_engine;

template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>&, const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

template <typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
const sycl::stream&
operator<<(const sycl::stream&, const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
std::basic_istream<_CharT, _Traits>&
operator>>(std::basic_istream<_CharT, _Traits>&, philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

template <typename _UIntType, std::size_t _W, std::size_t _N, std::size_t _R,
          oneapi::dpl::internal::element_type_t<_UIntType>... _Consts>
class philox_engine
{
  public:
    /* Types */
    using result_type = _UIntType;
    using scalar_type = oneapi::dpl::internal::element_type_t<result_type>;

  private:
    /* The size of the consts arrays */
    static constexpr std::size_t __array_size = _N / 2;

    /* Method for unpacking even and odd elements of input constants into an array */
    enum class __indices_offset : std::size_t
    {
        __even_indices = 0,
        __odd_indices = 1
    };
    template <__indices_offset _Offset, std::size_t... _Is>
    static constexpr auto
    __get_consts_by(std::index_sequence<_Is...>)
    {
        constexpr std::array __input_array{_Consts...};
        return std::array<scalar_type, sizeof...(_Is)>{__input_array[_Is * 2 + static_cast<std::size_t>(_Offset)]...};
    }

  public:
    /* Engine characteristics */
    static constexpr std::size_t word_size = _W;
    static constexpr std::size_t word_count = _N;
    static constexpr std::size_t round_count = _R;

    static_assert(_N == 2 || _N == 4, "parameter n must be 2 or 4");
    static_assert(sizeof...(_Consts) == _N, "the amount of consts must be equal to n");
    static_assert(_R > 0, "parameter r must be more than 0");
    static_assert(_W > 0 && _W <= std::numeric_limits<scalar_type>::digits,
                  "parameter w must satisfy 0 < w < std::numeric_limits<UIntType>::digits");
    static_assert(std::numeric_limits<scalar_type>::digits <= 64,
                  "size of the scalar UIntType (in case of sycl::vec<T, N> the size of T) must be less than 64 bits");
    static_assert(std::is_unsigned_v<scalar_type>, "UIntType must be unsigned type or vector of unsigned types");

    static constexpr std::array<scalar_type, __array_size> multipliers =
        __get_consts_by<__indices_offset::__even_indices>(std::make_index_sequence<__array_size>{});
    static constexpr std::array<scalar_type, __array_size> round_consts =
        __get_consts_by<__indices_offset::__odd_indices>(std::make_index_sequence<__array_size>{});

    static constexpr scalar_type
    min()
    {
        return 0;
    }

    static constexpr scalar_type
    max()
    {
        // equals to 2^w - 1
        return __in_mask;
    }

    static constexpr scalar_type default_seed = 20111115u;

    /* Constructors */
    philox_engine() : philox_engine(default_seed) {}

    explicit philox_engine(scalar_type __seed) { seed(__seed); }

    /* Seeding function */
    void
    seed(scalar_type __seed = default_seed)
    {
        __seed_internal(__seed & __in_mask);
    }

    /* Set the state to arbitrary position */
    void
    set_counter(const std::array<scalar_type, word_count>& __counter)
    {
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            // all counters are set in reverse order
            __state.__x[word_count - __i - 1] = __counter[__i] & __in_mask;
        }
        __state.__idx = word_count - 1;
    }

    /* Generating functions */
    result_type
    operator()()
    {
        return __generate_internal<oneapi::dpl::internal::type_traits_t<result_type>::num_elems>();
    }

    /* operator () overload for result portion generation */
    result_type
    operator()(unsigned int __random_nums)
    {
        return __generate_internal<oneapi::dpl::internal::type_traits_t<result_type>::num_elems>(__random_nums);
    }

    /* Shift the counter only forward relative to its current position */
    void
    discard(unsigned long long __z)
    {
        __discard_internal(__z);
    }

    /* Equality operators */
    friend bool
    operator==(const philox_engine& __eng1, const philox_engine& __eng2)
    {
        return (std::equal(__eng1.__state.__x.begin(), __eng1.__state.__x.end(), __eng2.__state.__x.begin()) &&
                std::equal(__eng1.__state.__k.begin(), __eng1.__state.__k.end(), __eng2.__state.__k.begin()) &&
                std::equal(__eng1.__state.__y.begin(), __eng1.__state.__y.end(), __eng2.__state.__y.begin()) &&
                __eng1.__state.__idx == __eng2.__state.__idx);
    }

    friend bool
    operator!=(const philox_engine& __eng1, const philox_engine& __eng2)
    {
        return !(__eng1 == __eng2);
    }

    /* Inserters and extractors */
    template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
              oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
    friend std::basic_ostream<_CharT, _Traits>&
    operator<<(std::basic_ostream<_CharT, _Traits>&, const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

    template <typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
              oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
    friend const sycl::stream&
    operator<<(const sycl::stream&, const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

    template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
              oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
    friend std::basic_istream<_CharT, _Traits>&
    operator>>(std::basic_istream<_CharT, _Traits>&, philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>&);

  private:
    /* Internal generator state */
    struct __state_type
    {
        std::array<scalar_type, word_count> __x;     // counters
        std::array<scalar_type, word_count / 2> __k; // keys
        std::array<scalar_type, word_count> __y;     // results
        scalar_type __idx;                           // index
    } __state;

    /* __word_mask<_WordSize> - scalar_type with the low _WordSize bits set */
    template <std::size_t _WordSize, typename = std::enable_if_t<_WordSize != 0>>
    static constexpr scalar_type __word_mask = ~scalar_type(0) >>
                                               (std::numeric_limits<scalar_type>::digits - _WordSize);

    /* Processing mask */
    static constexpr auto __in_mask = __word_mask<word_size>;

    void
    __seed_internal(scalar_type __seed)
    {
        // set to zero counters and results
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            __state.__x[__i] = 0;
            __state.__y[__i] = 0;
        }
        // 0th key element is set as seed, others are 0
        __state.__k[0] = __seed & __in_mask;
        for (std::size_t __i = 1; __i < (word_count / 2); ++__i)
        {
            __state.__k[__i] = 0;
        }

        __state.__idx = word_count - 1;
    }

    /* Increment counter by 1 */
    void
    __increase_counter_internal()
    {
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            __state.__x[__i] = (__state.__x[__i] + 1) & __in_mask;
            if (__state.__x[__i])
            {
                return;
            }
        }
    }

    /* Increment counter by an arbitrary __z */
    void
    __increase_counter_internal(unsigned long long __z)
    {
        unsigned long long __carry = 0;
        unsigned long long __ctr_inc = __z;

        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            scalar_type __initial_x = __state.__x[__i];
            __state.__x[__i] = (__initial_x + __ctr_inc + __carry) & __in_mask;

            __carry = 0;
            // check overflow of the current chunk
            if (__state.__x[__i] < __initial_x)
            {
                __carry = 1;
            }

            // select high chunk and shift for addition with the next counter chunk
            __ctr_inc = (word_size == std::numeric_limits<unsigned long long>::digits)
                            ? 0 // should be added only once for 64-bit word_size
                            : __ctr_inc >> word_size;
        }
    }

    /* Decrement counter by 1 */
    void
    __decrease_counter_internal()
    {
        for (std::size_t __i = 0; __i < word_count; ++__i)
        {
            if (__state.__x[__i])
            {
                __state.__x[__i] = (__state.__x[__i] - 1) & __in_mask;
                return;
            }

            /* Borrow for zero counter chunk */
            __state.__x[__i] = __in_mask;
        }
    }

    /* __generate_internal() specified for sycl_vec output 
       and overload for result portion generation */
    template <unsigned int _VecSize>
    std::enable_if_t<(_VecSize > 0), result_type>
    __generate_internal(unsigned int __random_nums)
    {
        if (__random_nums >= _VecSize)
            return operator()();

        result_type __loc_result;
        for (int __elm_count = 0; __elm_count < __random_nums; ++__elm_count)
        {
            ++__state.__idx;

            // check if buffer is empty
            if (__state.__idx == word_count)
            {
                __philox_kernel();
                __increase_counter_internal();
                __state.__idx = 0;
            }
            __loc_result[__elm_count] = __state.__y[__state.__idx];
        }

        return __loc_result;
    }

    /* __generate_internal() specified for sycl_vec output */
    template <unsigned int _VecSize>
    std::enable_if_t<(_VecSize > 0), result_type>
    __generate_internal()
    {
        result_type __loc_result;
        for (int __elm_count = 0; __elm_count < _VecSize; ++__elm_count)
        {
            ++__state.__idx;

            // check if buffer is empty
            if (__state.__idx == word_count)
            {
                __philox_kernel();
                __increase_counter_internal();
                __state.__idx = 0;
            }
            __loc_result[__elm_count] = __state.__y[__state.__idx];
        }

        return __loc_result;
    }

    /* __generate_internal() specified for a scalar output */
    template <unsigned int _VecSize>
    std::enable_if_t<(_VecSize == 0), result_type>
    __generate_internal()
    {
        ++__state.__idx;
        if (__state.__idx == word_count)
        {
            __philox_kernel();
            __increase_counter_internal();
            __state.__idx = 0;
        }

        return __state.__y[__state.__idx];
    }

    void
    __discard_internal(unsigned long long __z)
    {
        std::uint32_t __available_in_buffer = word_count - 1 - __state.__idx;
        if (__z <= __available_in_buffer)
        {
            __state.__idx += __z;
        }
        else
        {
            __z -= __available_in_buffer;
            int __tail = __z % word_count;
            if (__tail == 0)
            {
                __increase_counter_internal(__z / word_count);
                __state.__idx = word_count - 1;
            }
            else
            {
                if (__z > word_count)
                {
                    __increase_counter_internal((__z - 1) / word_count);
                }
                __philox_kernel();
                __increase_counter_internal();
                __state.__idx = __tail - 1;
            }
        }
    }

    /* Internal generation Philox kernel */
    void
    __philox_kernel()
    {
        if constexpr (word_count == 2)
        {
            scalar_type& __v0 = __state.__y[0];
            scalar_type& __v1 = __state.__y[1];
            __v0 = __state.__x[0];
            __v1 = __state.__x[1];
            scalar_type __k0 = __state.__k[0];
            for (std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = __mulhilo(__v0, multipliers[0]);
                __v0 = __hi0 ^ __k0 ^ __v1;
                __v1 = __lo0;
                __k0 = (__k0 + round_consts[0]) & __in_mask;
            }
        }
        else if constexpr (word_count == 4)
        {
            scalar_type& __v0 = __state.__y[2];
            scalar_type& __v1 = __state.__y[1];
            scalar_type& __v2 = __state.__y[0];
            scalar_type& __v3 = __state.__y[3];

            // permute __x to V
            __v2 = __state.__x[0];
            __v1 = __state.__x[1];
            __v0 = __state.__x[2];
            __v3 = __state.__x[3];
            scalar_type __k0 = __state.__k[0];
            scalar_type __k1 = __state.__k[1];
            for (std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = __mulhilo(__v0, multipliers[0]);
                auto [__hi1, __lo1] = __mulhilo(__v2, multipliers[1]);
                __v2 = __hi0 ^ __v1 ^ __k0;
                __v1 = __lo0;
                __v0 = __hi1 ^ __v3 ^ __k1;
                __v3 = __lo1;
                __k0 = (__k0 + round_consts[0]) & __in_mask;
                __k1 = (__k1 + round_consts[1]) & __in_mask;
            }
        }
    }

    /* Returns the word_size high and word_size low
       bits of the 2*word_size-bit product of __a and __b */
    static std::pair<scalar_type, scalar_type>
    __mulhilo(scalar_type __a, scalar_type __b)
    {
        scalar_type __res_hi, __res_lo;

        /* multiplication fits standard types */
        if constexpr (word_size <= 32)
        {
            std::uint_fast64_t __mult_result = (std::uint_fast64_t)__a * (std::uint_fast64_t)__b;
            __res_hi = __mult_result >> word_size;
            __res_lo = __mult_result;
        }
        /* pen-and-pencil multiplication by 32-bit chunks */
        else if constexpr (word_size > 32)
        {
            constexpr std::size_t __chunk_size = 32;
            __res_lo = __a * __b;

            scalar_type __x0 = __a & __word_mask<__chunk_size>;
            scalar_type __x1 = __a >> __chunk_size;
            scalar_type __y0 = __b & __word_mask<__chunk_size>;
            scalar_type __y1 = __b >> __chunk_size;

            scalar_type __p11 = __x1 * __y1;
            scalar_type __p01 = __x0 * __y1;
            scalar_type __p10 = __x1 * __y0;
            scalar_type __p00 = __x0 * __y0;

            /* addition of three 32-bit values to get the carry for the hi part */
            scalar_type __carry_hi =
                ((__p10 & __word_mask<__chunk_size>)+(__p00 >> __chunk_size) + (__p01 & __word_mask<__chunk_size>)) >>
                __chunk_size;

            /* 64-bit product + two 32-bit values + carry from the lo part */
            __res_hi = (__p11 + (__p01 >> __chunk_size) + (__p10 >> __chunk_size) + __carry_hi);

            /* for w!=64 the result should be concatenated with the lo part */
            __res_hi =
                (word_size == std::numeric_limits<scalar_type>::digits)
                    ? __res_hi
                    : __res_hi << (std::numeric_limits<scalar_type>::digits - word_size) | (__res_lo >> word_size);
        }

        return {__res_hi & __in_mask, __res_lo & __in_mask};
    }
};

template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>& __engine)
{
    oneapi::dpl::internal::save_stream_flags<_CharT, _Traits> __flags(__os);

    __os.setf(std::ios_base::dec | std::ios_base::left);
    _CharT __sp = __os.widen(' ');
    __os.fill(__sp);

    for (auto __k_elm : __engine.__state.__k)
    {
        __os << __k_elm << __sp;
    }
    for (auto __x_elm : __engine.__state.__x)
    {
        __os << __x_elm << __sp;
    }
    __os << __engine.__state.__idx;

    return __os;
}

template <typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
const sycl::stream&
operator<<(const sycl::stream& __os, const philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>& __engine)
{
    for (auto __k_elm : __engine.__state.__k)
    {
        __os << __k_elm << ' ';
    }
    for (auto __x_elm : __engine.__state.__x)
    {
        __os << __x_elm << ' ';
    }
    __os << __engine.__state.__idx;

    return __os;
}

template <typename _CharT, typename _Traits, typename _UIntTypeP, std::size_t _Wp, std::size_t _Np, std::size_t _Rp,
          oneapi::dpl::internal::element_type_t<_UIntTypeP>... _ConstsP>
std::basic_istream<_CharT, _Traits>&
operator>>(std::basic_istream<_CharT, _Traits>& __is, philox_engine<_UIntTypeP, _Wp, _Np, _Rp, _ConstsP...>& __engine)
{
    oneapi::dpl::internal::save_stream_flags<_CharT, _Traits> __flags(__is);

    __is.setf(std::ios_base::dec);

    /* Number of elements in the state (__k, __x and __idx) */
    constexpr std::size_t __state_size = _Np / 2 + _Np + 1;

    std::array<oneapi::dpl::internal::element_type_t<_UIntTypeP>, __state_size> __tmp_inp;
    for (std::size_t __i = 0; __i < __state_size; ++__i)
    {
        __is >> __tmp_inp[__i];
    }

    if (!__is.fail())
    {
        int __inp_itr = 0;
        for (std::size_t __i = 0; __i < _Np / 2; ++__i, ++__inp_itr)
        {
            __engine.__state.__k[__i] = __tmp_inp[__inp_itr];
        }
        for (std::size_t __i = 0; __i < _Np; ++__i, ++__inp_itr)
        {
            __engine.__state.__x[__i] = __tmp_inp[__inp_itr];
        }
        __engine.__state.__idx = __tmp_inp[__inp_itr];

        /* Counter is incremented right after the generation of Yi - to restore the unused sequence Yi, the counter has to be decremented */
        if (__engine.__state.__idx != _Np - 1)
        {
            __engine.__decrease_counter_internal();
            /* setup Yi */
            __engine.__philox_kernel();
        }
    }

    return __is;
}

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H
