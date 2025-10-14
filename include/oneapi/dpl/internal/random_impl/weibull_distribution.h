// -*- C++ -*-
//===-- weibull_distribution.h ------------------------------------------===//
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
// Public header file provides implementation for Weibull Distribution

#ifndef _ONEDPL_WEIBULL_DISTRIBUTION_H
#define _ONEDPL_WEIBULL_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{
template <class _RealType = double>
class weibull_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;
    class param_type
    {
      public:
        using distribution_type = weibull_distribution<result_type>;
        param_type() : param_type(scalar_type{1.0}) {}
        explicit param_type(scalar_type a, scalar_type b = scalar_type{1.0}) : a_(a), b_(b) {}
        scalar_type
        a() const
        {
            return a_;
        }
        scalar_type
        b() const
        {
            return b_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.a_ == p2.a_ && p1.b_ == p2.b_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        scalar_type a_;
        scalar_type b_;
    };

    // Constructors
    weibull_distribution() : weibull_distribution(scalar_type{1.0}) {}
    explicit weibull_distribution(scalar_type __a, scalar_type __b = scalar_type{1.0}) : a_(__a), b_(__b) {}
    explicit weibull_distribution(const param_type& __params) : a_(__params.a()), b_(__params.b()) {}

    // Reset function
    void
    reset()
    {
    }

    // Property functions
    scalar_type
    a() const
    {
        return a_;
    }

    scalar_type
    b() const
    {
        return b_;
    }

    param_type
    param() const
    {
        return param_type(a_, b_);
    }

    void
    param(const param_type& __params)
    {
        a_ = __params.a();
        b_ = __params.b();
    }

    scalar_type
    min() const
    {
        return scalar_type{};
    }

    scalar_type
    max() const
    {
        return std::numeric_limits<scalar_type>::max();
    }

    // Generate functions
    template <class _Engine>
    result_type
    operator()(_Engine& __engine)
    {
        return operator()<_Engine>(__engine, param_type(a_, b_));
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params)
    {
        return generate<size_of_type_, _Engine>(__engine, __params);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, unsigned int __random_nums)
    {
        return operator()<_Engine>(__engine, param_type(a_, b_), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

    friend bool
    operator==(const weibull_distribution& __x, const weibull_distribution& __y)
    {
        return __x.param() == __y.param();
    }

    friend bool
    operator!=(const weibull_distribution& __x, const weibull_distribution& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os, const weibull_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__os);

        __os.setf(std::ios_base::dec | std::ios_base::left);
        CharT __sp = __os.widen(' ');
        __os.fill(__sp);

        return __os << __d.a() << __sp << __d.b();
    }

    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const weibull_distribution& __d)
    {
        return __os << __d.a() << ' ' << __d.b();
    }

    template <class CharT, class Traits>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, weibull_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__is);

        __is.setf(std::ios_base::dec);

        weibull_distribution::scalar_type __a;
        weibull_distribution::scalar_type __b;

        if (__is >> __a >> __b)
            __d.param(weibull_distribution::param_type(__a, __b));

        return __is;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Static asserts
    static_assert(::std::is_floating_point_v<scalar_type>,
                  "oneapi::dpl::weibull_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type a_;
    scalar_type b_;

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr != 0), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        return generate_vec<_Ndistr, _Engine>(__engine, __params);
    }

    // Specialization of the scalar generation
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr == 0), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        oneapi::dpl::uniform_real_distribution<scalar_type> __u;
        return __params.b() * sycl::pow(-sycl::log(scalar_type{1.0} - __u(__engine)), scalar_type{1.0} / __params.a());
    }

    // Specialization of the vector generation with size = [1; 2; 3]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N <= 3), result_type>
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        return generate_n_elems<_Engine>(__engine, __params, __N);
    }

    // Specialization of the vector generation with size = [4; 8; 16]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N > 3), result_type>
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        oneapi::dpl::uniform_real_distribution<sycl::vec<scalar_type, __N>> __distr;
        return __params.b() *
               sycl::pow(-sycl::log(scalar_type{1.0} - __distr(__engine)), result_type{1.0} / __params.a());
    }

    // Implementation for the N vector's elements generation
    template <class _Engine>
    result_type
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res;
        oneapi::dpl::uniform_real_distribution<scalar_type> __u;
        scalar_type __tmp = scalar_type{1.0} / __params.a();
        for (unsigned int i = 0; i < __N; i++)
            __res[i] = __params.b() * sycl::pow(-sycl::log(scalar_type{1.0} - __u(__engine)), __tmp);
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr != 0), result_type>
    result_portion_internal(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __part_vec;
        if (__N == 0)
            return __part_vec;
        else if (__N >= _Ndistr)
            return operator()(__engine, __params);

        __part_vec = generate_n_elems(__engine, __params, __N);
        return __part_vec;
    }
};
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_WEIBULL_DISTRIBUTION_H
