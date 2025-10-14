// -*- C++ -*-
//===-- exponential_distribution.h ----------------------------------------===//
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
// Public header file provides implementation for Exponential Distribution

#ifndef _ONEDPL_EXPONENTIAL_DISTRIBUTION_H
#define _ONEDPL_EXPONENTIAL_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{
template <class _RealType = double>
class exponential_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;
    class param_type
    {
      public:
        using distribution_type = exponential_distribution<result_type>;
        param_type() : param_type(scalar_type{1.0}) {}
        explicit param_type(scalar_type lambda) : lambda_(lambda) {}
        scalar_type
        lambda() const
        {
            return lambda_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.lambda_ == p2.lambda_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        scalar_type lambda_;
    };

    // Constructors
    exponential_distribution() : exponential_distribution(scalar_type{1.0}) {}
    explicit exponential_distribution(scalar_type __lambda) : lambda_(__lambda) {}
    explicit exponential_distribution(const param_type& __params) : lambda_(__params.lambda()) {}

    // Reset function
    void
    reset()
    {
    }

    // Property functions
    scalar_type
    lambda() const
    {
        return lambda_;
    }

    param_type
    param() const
    {
        return param_type(lambda_);
    }

    void
    param(const param_type& __params)
    {
        lambda_ = __params.lambda();
    }

    scalar_type
    min() const
    {
        return scalar_type{};
    }

    scalar_type
    max() const
    {
        return ::std::numeric_limits<scalar_type>::infinity();
    }

    // Generate functions
    template <class _Engine>
    result_type
    operator()(_Engine& __engine)
    {
        return operator()<_Engine>(__engine, param_type(lambda_));
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
        return operator()<_Engine>(__engine, param_type(lambda_), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

    friend bool
    operator==(const exponential_distribution& __x, const exponential_distribution& __y)
    {
        return __x.lambda_ == __y.lambda_;
    }

    friend bool
    operator!=(const exponential_distribution& __x, const exponential_distribution& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os, const exponential_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__os);

        __os.setf(std::ios_base::dec | std::ios_base::left);
        __os.fill(__os.widen(' '));

        return __os << __d.lambda();
    }

    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const exponential_distribution& __d)
    {
        return __os << __d.lambda();
    }

    template <class CharT, class Traits>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, exponential_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__is);

        __is.setf(std::ios_base::dec);

        scalar_type __lambda;

        if (__is >> __lambda)
            __d.param(exponential_distribution::param_type(__lambda));

        return __is;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Static asserts
    static_assert(::std::is_floating_point_v<scalar_type>,
                  "oneapi::dpl::exponential_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type lambda_;

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
        result_type __res;
        oneapi::dpl::uniform_real_distribution<scalar_type> __u;
        __res = -sycl::log(scalar_type{1.0} - __u(__engine)) / __params.lambda();
        return __res;
    }

    // Specialization of the vector generation  with size = [1; 2; 3]
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
        oneapi::dpl::uniform_real_distribution<result_type> __u;
        result_type __res;
        __res = __u(__engine);
        __res = -sycl::log(scalar_type{1.0} - __res) / __params.lambda();
        return __res;
    }

    // Implementation for the N vector's elements generation
    template <class _Engine>
    result_type
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res;
        oneapi::dpl::uniform_real_distribution<scalar_type> __u;
        for (unsigned int i = 0; i < __N; i++)
        {
            __res[i] = -sycl::log(scalar_type{1.0} - __u(__engine)) / __params.lambda();
        }
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

#endif // _ONEDPL_EXPONENTIAL_DISTRIBUTION_H
