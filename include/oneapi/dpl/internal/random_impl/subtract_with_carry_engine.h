// -*- C++ -*-
//===-- subtract_with_carry_engine.h --------------------------------------===//
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
// Public header file provides implementation for Subtract with Carry Engine

#ifndef _ONEDPL_SUBTRACT_WITH_CARRY_ENGINE_H
#define _ONEDPL_SUBTRACT_WITH_CARRY_ENGINE_H

namespace oneapi
{
namespace dpl
{

template <class _UIntType, size_t _W, size_t _S, size_t _R>
class subtract_with_carry_engine;

template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
::std::basic_ostream<CharT, Traits>&
operator<<(::std::basic_ostream<CharT, Traits>&, const subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

template <class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
const sycl::stream&
operator<<(const sycl::stream&, const subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
::std::basic_istream<CharT, Traits>&
operator>>(::std::basic_istream<CharT, Traits>&, subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

template <class _UIntType, size_t _W, size_t _S, size_t _R>
class subtract_with_carry_engine
{
  public:
    // Engine types
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    // Engine characteristics
    static constexpr size_t word_size = _W;
    static constexpr size_t short_lag = _S;
    static constexpr size_t long_lag = _R;
    static constexpr scalar_type
    min()
    {
        return 0u;
    }
    static constexpr scalar_type
    max()
    {
        return (static_cast<scalar_type>(1u) << word_size) - static_cast<scalar_type>(1u);
    }
    static constexpr scalar_type default_seed = 19780503u;

    // Constructors
    subtract_with_carry_engine() : subtract_with_carry_engine(default_seed){};

    explicit subtract_with_carry_engine(scalar_type __seed, unsigned long long __offset = 0)
    {
        // Engine initialization
        init(__seed);

        // Discard offset
        discard(__offset);
    }

    // Seeding function
    void
    seed(scalar_type __seed = default_seed)
    {
        // Engine initialization
        init(__seed);
    }

    // Discard procedure
    void
    discard(unsigned long long __num_to_skip)
    {
        if (!__num_to_skip)
            return;

        for (; __num_to_skip > 0; --__num_to_skip)
            generate_internal_scalar();
    }

    // operator () returns bits of engine recurrence
    result_type
    operator()()
    {
        return generate_internal<internal::type_traits_t<result_type>::num_elems>();
    }

    // operator () overload for result portion generation
    result_type
    operator()(unsigned int __random_nums)
    {
        return result_portion_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
    }

    friend bool
    operator==(const subtract_with_carry_engine& __x, const subtract_with_carry_engine& __y)
    {
        if (__x.c_ != __y.c_)
            return false;
        if (__x.i_ == __y.i_)
            return std::equal(__x.x_, __x.x_ + _R, __y.x_);
        if (__x.i_ == 0 || __y.i_ == 0)
        {
            size_t __j = std::min(_R - __x.i_, _R - __y.i_);
            if (!std::equal(__x.x_ + __x.i_, __x.x_ + __x.i_ + __j, __y.x_ + __y.i_))
                return false;
            if (__x.i_ == 0)
                return std::equal(__x.x_ + __j, __x.x_ + _R, __y.x_);
            return std::equal(__x.x_, __x.x_ + (_R - __j), __y.x_ + __j);
        }
        if (__x.i_ < __y.i_)
        {
            size_t __j = _R - __y.i_;
            if (!std::equal(__x.x_ + __x.i_, __x.x_ + (__x.i_ + __j), __y.x_ + __y.i_))
                return false;
            if (!std::equal(__x.x_ + (__x.i_ + __j), __x.x_ + _R, __y.x_))
                return false;
            return std::equal(__x.x_, __x.x_ + __x.i_, __y.x_ + (_R - (__x.i_ + __j)));
        }
        size_t __j = _R - __x.i_;
        if (!std::equal(__y.x_ + __y.i_, __y.x_ + (__y.i_ + __j), __x.x_ + __x.i_))
            return false;
        if (!std::equal(__y.x_ + (__y.i_ + __j), __y.x_ + _R, __x.x_))
            return false;
        return std::equal(__y.x_, __y.x_ + __y.i_, __x.x_ + (_R - (__y.i_ + __j)));
    }

    friend bool
    operator!=(const subtract_with_carry_engine& __x, const subtract_with_carry_engine& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>&, const subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

    template <class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
    friend const sycl::stream&
    operator<<(const sycl::stream&, const subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

    template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>&, subtract_with_carry_engine<__UIntType, __W, __S, __R>&);

  private:
    // Static asserts
    static_assert((0 < _S) && (_S < _R), "oneapi::dpl::subtract_with_carry_engine. Error: unsupported parameters");

    static_assert((0 < _W) && (_W <= std::numeric_limits<scalar_type>::digits),
                  "oneapi::dpl::subtract_with_carry_engine. Error: unsupported parameters");
    // Initialization function
    void
    init(scalar_type __seed)
    {
        linear_congruential_engine<scalar_type, 40014u, 0u, 2147483563u> __engine(__seed == 0 ? default_seed : __seed);

        for (size_t __i = 0; __i < long_lag; ++__i)
        {
            x_[__i] = static_cast<scalar_type>(__engine() & max());

            if (word_size > 32)
            {
                ::std::uint64_t __tmp = (::std::uint64_t(__engine()) << 32);
                x_[__i] = x_[__i] + __tmp;
            }

            x_[__i] &= max();
        }
        c_ = x_[long_lag - 1] == 0;
    }

    // Function for state adjustment
    scalar_type
    generate_internal_scalar()
    {
        const scalar_type& __xs = x_[(i_ + (long_lag - short_lag)) % long_lag];
        scalar_type& __xr = x_[i_];
        scalar_type __new_c = c_ == 0 ? __xs < __xr : __xs != 0 ? __xs <= __xr : 1;
        x_[i_] = __xr = (__xs - __xr - c_) & max();
        c_ = __new_c;
        i_ = (i_ + 1) % long_lag;
        return __xr;
    };

    // Generate implementation
    template <int _N>
    ::std::enable_if_t<(_N == 0), result_type>
    generate_internal()
    {
        return generate_internal_scalar();
    }

    template <int _N>
    ::std::enable_if_t<(_N > 0), result_type>
    generate_internal()
    {
        result_type __res;
        for (int __i = 0; __i < _N; ++__i)
        {
            __res[__i] = generate_internal_scalar();
        }
        return __res;
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
        {
            __part_vec[__i] = generate_internal_scalar();
        }

        return __part_vec;
    }

    scalar_type x_[long_lag];
    scalar_type c_;
    size_t i_ = 0;
};

template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
::std::basic_ostream<CharT, Traits>&
operator<<(::std::basic_ostream<CharT, Traits>& __os, const subtract_with_carry_engine<__UIntType, __W, __S, __R>& __e)
{
    internal::save_stream_flags<CharT, Traits> __flags(__os);

    __os.setf(std::ios_base::dec | std::ios_base::left);
    CharT __sp = __os.widen(' ');
    __os.fill(__sp);

    __os << __e.x_[__e.i_];
    for (size_t j = __e.i_ + 1; j < __R; ++j)
        __os << __sp << __e.x_[j];
    for (size_t j = 0; j < __e.i_; ++j)
        __os << __sp << __e.x_[j];

    __os << __sp << __e.c_;

    return __os;
}

template <class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
const sycl::stream&
operator<<(const sycl::stream& __os, const subtract_with_carry_engine<__UIntType, __W, __S, __R>& __e)
{
    __os << __e.x_[__e.i_];
    for (size_t j = __e.i_ + 1; j < __R; ++j)
        __os << ' ' << __e.x_[j];
    for (size_t j = 0; j < __e.i_; ++j)
        __os << ' ' << __e.x_[j];

    __os << ' ' << __e.c_;

    return __os;
}

template <class CharT, class Traits, class __UIntType, std::size_t __W, std::size_t __S, std::size_t __R>
::std::basic_istream<CharT, Traits>&
operator>>(::std::basic_istream<CharT, Traits>& __is, subtract_with_carry_engine<__UIntType, __W, __S, __R>& __e)
{
    internal::save_stream_flags<CharT, Traits> __flags(__is);

    __is.setf(std::ios_base::dec);

    __UIntType __t[__R + 1];
    for (size_t i = 0; i < __R + 1; ++i)
        __is >> __t[i];
    if (!__is.fail())
    {
        for (size_t i = 0; i < __R; ++i)
            __e.x_[i] = __t[i];
        __e.c_ = __t[__R];
        __e.i_ = 0;
    }

    return __is;
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SUBTRACT_WITH_CARRY_ENGINE_H
