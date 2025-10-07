// -*- C++ -*-
//===-- linear_congruential_engine.h --------------------------------------===//
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
//
// Abstract:
//
// Public header file provides implementation for Linear Congruential Engine

#ifndef _ONEDPL_LINEAR_CONGRUENTIAL_ENGINE_H
#define _ONEDPL_LINEAR_CONGRUENTIAL_ENGINE_H

namespace oneapi
{
namespace dpl
{

template <class _UIntType, internal::element_type_t<_UIntType> _A, internal::element_type_t<_UIntType> _C,
          internal::element_type_t<_UIntType> _M>
class linear_congruential_engine;

template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
          internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
::std::basic_ostream<CharT, Traits>&
operator<<(::std::basic_ostream<CharT, Traits>&, const linear_congruential_engine<__UIntType, __A, __C, __M>&);

template <class __UIntType, internal::element_type_t<__UIntType> __A, internal::element_type_t<__UIntType> __C,
          internal::element_type_t<__UIntType> __M>
const sycl::stream&
operator<<(const sycl::stream&, const linear_congruential_engine<__UIntType, __A, __C, __M>&);

template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
          internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
::std::basic_istream<CharT, Traits>&
operator>>(::std::basic_istream<CharT, Traits>&, linear_congruential_engine<__UIntType, __A, __C, __M>&);

template <class _UIntType, internal::element_type_t<_UIntType> _A, internal::element_type_t<_UIntType> _C,
          internal::element_type_t<_UIntType> _M>
class linear_congruential_engine
{
  public:
    // Engine types
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    // Engine characteristics
    static constexpr scalar_type multiplier = _A;
    static constexpr scalar_type increment = _C;
    static constexpr scalar_type modulus = _M;
    static constexpr scalar_type
    min()
    {
        return (increment == static_cast<scalar_type>(0u)) ? static_cast<scalar_type>(1u)
                                                           : static_cast<scalar_type>(0u);
    }
    static constexpr scalar_type
    max()
    {
        return modulus - static_cast<scalar_type>(1u);
    }
    static constexpr scalar_type default_seed = static_cast<scalar_type>(1u);

    // Constructors
    linear_congruential_engine() : linear_congruential_engine(default_seed){};

    explicit linear_congruential_engine(scalar_type __seed, unsigned long long __offset = 0)
    {
        init<internal::type_traits_t<result_type>::num_elems>(__seed);
        discard(__offset);
    }

    // Seeding function
    void
    seed(scalar_type __seed = default_seed)
    {
        // Engine initialization
        init<internal::type_traits_t<result_type>::num_elems>(__seed);
    }

    // Discard procedure
    void
    discard(unsigned long long __num_to_skip)
    {
        // Skipping sequence
        if (__num_to_skip == 0)
            return;
        constexpr bool __flag = (increment == 0) && (modulus < ::std::numeric_limits<::std::uint32_t>::max()) &&
                                (multiplier < ::std::numeric_limits<::std::uint32_t>::max());
        skip_seq<internal::type_traits_t<result_type>::num_elems, __flag>(__num_to_skip);
    }

    // operator () returns bits of engine recurrence
    result_type
    operator()()
    {
        result_type __state_tmp = state_;
        unsigned long long __discard_num = (internal::type_traits_t<result_type>::num_elems == 0)
                                               ? 1
                                               : internal::type_traits_t<result_type>::num_elems;
        discard(__discard_num);
        return __state_tmp;
    }

    // operator () overload for result portion generation
    result_type
    operator()(unsigned int __random_nums)
    {
        return result_portion_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
    }

    friend bool
    operator==(const linear_congruential_engine& __x, const linear_congruential_engine& __y)
    {
        return __x.state_ == __y.state_;
    }

    friend bool
    operator!=(const linear_congruential_engine& __x, const linear_congruential_engine& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
              internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os,
               const linear_congruential_engine<__UIntType, __A, __C, __M>& __e);

    template <class __UIntType, internal::element_type_t<__UIntType> __A, internal::element_type_t<__UIntType> __C,
              internal::element_type_t<__UIntType> __M>
    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const linear_congruential_engine<__UIntType, __A, __C, __M>& __e);

    template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
              internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, linear_congruential_engine<__UIntType, __A, __C, __M>& __e);

  private:
    // Static asserts
    static_assert(((_M == 0) || ((_A < _M) && (_C < _M))),
                  "oneapi::dpl::linear_congruential_engine. Error: unsupported parameters");

    // Function for state adjustment
    scalar_type
    mod_scalar(scalar_type __state_input)
    {
        ::std::uint64_t __mult = multiplier, __mod = modulus, __inc = increment;
        return static_cast<scalar_type>((__mult * ::std::uint64_t(__state_input) + __inc) % __mod);
    }

    // Initialization function
    template <int _N = 0>
    ::std::enable_if_t<(_N == 0)>
    init(scalar_type __seed)
    {
        if ((increment % modulus == 0) && (__seed % modulus == 0))
        {
            state_ = default_seed;
        }
        else
        {
            state_ = __seed;
        }
        state_ = mod_scalar(state_);
    }

    template <int _N = 0>
    ::std::enable_if_t<(_N > 0)>
    init(scalar_type __seed)
    {
        if ((increment % modulus == 0) && (__seed % modulus == 0))
        {
            state_[0] = default_seed;
        }
        else
        {
            state_[0] = __seed;
        }

        state_[0] = mod_scalar(state_[0]);

        for (int __i = 1u; __i < _N; ++__i)
            state_[__i] = mod_scalar(state_[__i - 1u]);
    }

    // Internal function for calculate degrees of multiplier
    scalar_type
    pow_mult_n(unsigned long long __num_to_skip)
    {
        ::std::uint64_t __a2;
        ::std::uint64_t __mod = static_cast<::std::uint64_t>(modulus);
        ::std::uint64_t __a = static_cast<::std::uint64_t>(multiplier);
        scalar_type __r = 1;

        do
        {
            if (__num_to_skip & 1)
            {
                __a2 = static_cast<::std::uint64_t>(__r) * __a;
                __r = static_cast<scalar_type>(__a2 % __mod);
            }

            __num_to_skip >>= 1;
            __a2 = __a * __a;
            __a = __a2 % __mod;

        } while (__num_to_skip);

        return __r;
    }

    // Internal function which is used in discard procedure
    // _FLAG - is flag that used for optimizations
    // if _FLAG == true in this case we can used optimized versions of skip_seq
    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N == 0) && (_FLAG == false)>
    skip_seq(unsigned long long __num_to_skip)
    {
        for (; __num_to_skip > 0; --__num_to_skip)
            state_ = mod_scalar(state_);
    }

    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N == 1) && (_FLAG == false)>
    skip_seq(unsigned long long __num_to_skip)
    {
        for (; __num_to_skip > 0; --__num_to_skip)
            state_[0] = mod_scalar(state_[0]);
    }

    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N > 1) && (_FLAG == false)>
    skip_seq(unsigned long long __num_to_skip)
    {
        for (; __num_to_skip > 0; --__num_to_skip)
        {
            for (int __i = 0; __i < (_N - 1); ++__i)
            {
                state_[__i] = state_[__i + 1];
            }
            state_[_N - 1] = mod_scalar(state_[_N - 2]);
        }
    }

    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N == 0) && (_FLAG == true)>
    skip_seq(unsigned long long __num_to_skip)
    {
        ::std::uint64_t __mod = modulus;
        ::std::uint64_t __mult = pow_mult_n(__num_to_skip);
        state_ = static_cast<scalar_type>((__mult * static_cast<::std::uint64_t>(state_)) % __mod);
    }

    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N == 1) && (_FLAG == true)>
    skip_seq(unsigned long long __num_to_skip)
    {
        ::std::uint64_t __mod = modulus;
        ::std::uint64_t __mult = pow_mult_n(__num_to_skip);
        state_[0] = static_cast<scalar_type>((__mult * static_cast<::std::uint64_t>(state_[0])) % __mod);
    }

    template <int _N = 0, bool _FLAG = false>
    ::std::enable_if_t<(_N > 1) && (_FLAG == true)>
    skip_seq(unsigned long long __num_to_skip)
    {
        ::std::uint64_t __mod = modulus;
        ::std::uint64_t __mult = pow_mult_n(__num_to_skip);
        state_ = ((__mult * state_.template convert<::std::uint64_t>()) % __mod).template convert<scalar_type>();
    }

    // result_portion implementation
    template <int _N>
    ::std::enable_if_t<(_N > 0), result_type>
    result_portion_internal(unsigned int __random_nums)
    {
        result_type __part_vec;

        if (__random_nums >= _N)
            return operator()();

        for (unsigned int __i = 0; __i < __random_nums; ++__i)
            __part_vec[__i] = state_[__i];

        discard(__random_nums);
        return __part_vec;
    }

    result_type state_;
};

template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
          internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
::std::basic_ostream<CharT, Traits>&
operator<<(::std::basic_ostream<CharT, Traits>& __os, const linear_congruential_engine<__UIntType, __A, __C, __M>& __e)
{
    internal::save_stream_flags<CharT, Traits> __flags(__os);

    __os.setf(std::ios_base::dec | std::ios_base::left);
    __os.fill(__os.widen(' '));

    return __os << __e.state_;
}

template <class __UIntType, internal::element_type_t<__UIntType> __A, internal::element_type_t<__UIntType> __C,
          internal::element_type_t<__UIntType> __M>
const sycl::stream&
operator<<(const sycl::stream& __os, const linear_congruential_engine<__UIntType, __A, __C, __M>& __e)
{
    return __os << __e.state_;
}

template <class CharT, class Traits, class __UIntType, internal::element_type_t<__UIntType> __A,
          internal::element_type_t<__UIntType> __C, internal::element_type_t<__UIntType> __M>
::std::basic_istream<CharT, Traits>&
operator>>(::std::basic_istream<CharT, Traits>& __is, linear_congruential_engine<__UIntType, __A, __C, __M>& __e)
{
    internal::save_stream_flags<CharT, Traits> __flags(__is);

    __is.setf(std::ios_base::dec);

    __UIntType __t;
    if (__is >> __t)
        __e.state_ = __t;

    return __is;
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_LINEAR_CONGRUENTIAL_ENGINE_H
