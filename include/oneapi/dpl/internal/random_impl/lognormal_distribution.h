// -*- C++ -*-
//===-- lognormal_distribution.h ------------------------------------------===//
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
// Public header file provides implementation for Lognormal Distribution

#ifndef _ONEDPL_LOGNORMAL_DISTRIBUTION_H
#define _ONEDPL_LOGNORMAL_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{
template <class _RealType = double>
class lognormal_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;
    class param_type
    {
      public:
        using distribution_type = lognormal_distribution<result_type>;
        param_type() : param_type(scalar_type{0.0}) {}
        explicit param_type(scalar_type m, scalar_type s = scalar_type{1.0}) : m_(m), s_(s) {}
        scalar_type
        m() const
        {
            return m_;
        }
        scalar_type
        s() const
        {
            return s_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.m_ == p2.m_ && p1.s_ == p2.s_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        scalar_type m_;
        scalar_type s_;
    };

    // Constructors
    lognormal_distribution() : lognormal_distribution(scalar_type{0.0}) {}
    explicit lognormal_distribution(scalar_type __mean, scalar_type __stddev = scalar_type{1.0}) : nd_(__mean, __stddev)
    {
    }
    explicit lognormal_distribution(const param_type& __params) : nd_(__params.m(), __params.s()) {}

    // Reset function
    void
    reset()
    {
        nd_.reset();
    }

    // Property functions
    scalar_type
    m() const
    {
        return nd_.mean();
    }

    scalar_type
    s() const
    {
        return nd_.stddev();
    }

    param_type
    param() const
    {
        return param_type(nd_.mean(), nd_.stddev());
    }

    void
    param(const param_type& __params)
    {
        nd_.param(normal_distr_param_type(__params.m(), __params.s()));
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
        return operator()<_Engine>(__engine, param_type(nd_.mean(), nd_.stddev()));
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
        return operator()<_Engine>(__engine, param_type(nd_.mean(), nd_.stddev()), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

    friend bool
    operator==(const lognormal_distribution& __x, const lognormal_distribution& __y)
    {
        return __x.nd_ == __y.nd_;
    }

    friend bool
    operator!=(const lognormal_distribution& __x, const lognormal_distribution& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os, const lognormal_distribution& __d)
    {
        return __os << __d.nd_;
    }

    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const lognormal_distribution& __d)
    {
        return __os << __d.nd_;
    }

    template <class CharT, class Traits>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, lognormal_distribution& __d)
    {
        return __is >> __d.nd_;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    using normal_distr =
        oneapi::dpl::normal_distribution<::std::conditional_t<(size_of_type_ <= 3), scalar_type, result_type>>;
    using normal_distr_param_type = typename normal_distr::param_type;

    // Static asserts
    static_assert(::std::is_floating_point_v<scalar_type>,
                  "oneapi::dpl::lognormal_distribution. Error: unsupported data type");

    // Distribution parameters
    normal_distr nd_;

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
        return sycl::exp(nd_(__engine, normal_distr_param_type(__params.m(), __params.s())));
    }

    // Specialization of the vector generation with size = [1; 2; 3]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N <= 3), result_type>
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        result_type __res;
        for (int i = 0; i < __N; i++)
            __res[i] = sycl::exp(nd_(__engine, normal_distr_param_type(__params.m(), __params.s())));
        return __res;
    }

    // Specialization of the vector generation with size = [4; 8; 16]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N > 3), result_type>
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        return sycl::exp(nd_(__engine, normal_distr_param_type(__params.m(), __params.s())));
    }

    // Implementation for the N vector's elements generation with size = [4; 8; 16]
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr > 3), result_type>
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res = nd_(__engine, normal_distr_param_type(__params.m(), __params.s()), __N);
        for (unsigned int i = 0; i < __N; i++)
            __res[i] = sycl::exp(__res[i]);
        return __res;
    }

    // Implementation for the N vector's elements generation with size = [1; 2; 3]
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr <= 3), result_type>
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res;
        for (unsigned int i = 0; i < __N; i++)
            __res[i] = sycl::exp(nd_(__engine, normal_distr_param_type(__params.m(), __params.s())));
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

        __part_vec = generate_n_elems<_Ndistr, _Engine>(__engine, __params, __N);
        return __part_vec;
    }
};
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_LOGNORMAL_DISTRIBUTION_H
