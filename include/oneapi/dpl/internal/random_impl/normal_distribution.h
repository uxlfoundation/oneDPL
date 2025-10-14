// -*- C++ -*-
//===-- normal_distribution.h ---------------------------------------------===//
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
// Public header file provides implementation for Normal Distribution

#ifndef _ONEDPL_NORMAL_DISTRIBUTION_H
#define _ONEDPL_NORMAL_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{
namespace internal
{

static const dp_union_t gaussian_dp_table[2]{
    {0x54442D18, 0x401921FB}, // Pi * 2
    {0x446D71C3, 0xC0874385}, // ln(0.494065e-323) = -744.440072
};

static const sp_union_t gaussian_sp_table[2]{
    {0x40C90FDB}, // Pi * 2
    {0xC2CE8ED0}, // ln(0.14012984e-44) = -103.278929
};

} // namespace internal

template <class _RealType = double>
class normal_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<result_type>;
    class param_type
    {
      public:
        using distribution_type = normal_distribution<result_type>;
        param_type() : param_type(scalar_type{0.0}) {}
        explicit param_type(scalar_type mean, scalar_type stddev = scalar_type{1.0}) : mean_(mean), stddev_(stddev) {}
        scalar_type
        mean() const
        {
            return mean_;
        }
        scalar_type
        stddev() const
        {
            return stddev_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.mean_ == p2.mean_ && p1.stddev_ == p2.stddev_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        scalar_type mean_;
        scalar_type stddev_;
    };

    // Constructors
    normal_distribution() : normal_distribution(scalar_type{0.0}) {}
    explicit normal_distribution(scalar_type __mean, scalar_type __stddev = scalar_type{1.0})
        : mean_(__mean), stddev_(__stddev)
    {
    }
    explicit normal_distribution(const param_type& __params) : mean_(__params.mean()), stddev_(__params.stddev()) {}

    // Reset function
    void
    reset()
    {
        flag_ = false;
    }

    // Property functions
    scalar_type
    mean() const
    {
        return mean_;
    }

    scalar_type
    stddev() const
    {
        return stddev_;
    }

    param_type
    param() const
    {
        return param_type(mean_, stddev_);
    }

    void
    param(const param_type& __params)
    {
        mean_ = __params.mean();
        stddev_ = __params.stddev();
    }

    scalar_type
    min() const
    {
        return -(::std::numeric_limits<scalar_type>::infinity());
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
        return operator()<_Engine>(__engine, param_type(mean_, stddev_));
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
        return operator()<_Engine>(__engine, param_type(mean_, stddev_), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

    friend bool
    operator==(const normal_distribution& __x, const normal_distribution& __y)
    {
        return __x.param() == __y.param();
    }

    friend bool
    operator!=(const normal_distribution& __x, const normal_distribution& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os, const normal_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__os);

        __os.setf(std::ios_base::dec | std::ios_base::left);
        CharT __sp = __os.widen(' ');
        __os.fill(__sp);

        __os << __d.mean() << __sp << __d.stddev() << __sp << __d.flag_;
        if (__d.flag_)
            __os << __sp << __d.saved_ln_ << __sp << __d.saved_u2_;

        return __os;
    }

    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const normal_distribution& __d)
    {
        __os << __d.mean() << ' ' << __d.stddev() << ' ' << __d.flag_;

        if (__d.flag_)
            __os << ' ' << __d.saved_ln_ << ' ' << __d.saved_u2_;

        return __os;
    }

    template <class CharT, class Traits>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, normal_distribution& __d)
    {
        using __scalar_type = normal_distribution::scalar_type;

        internal::save_stream_flags<CharT, Traits> __flags(__is);

        __is.setf(std::ios_base::dec);

        __scalar_type __mean;
        __scalar_type __stddev;
        bool __flag_;
        __scalar_type __saved_ln_;
        __scalar_type __saved_u2_;

        if (__is >> __mean >> __stddev)
            __d.param(param_type(__mean, __stddev));
        if (__is >> __flag_)
        {
            __d.flag_ = __flag_;
            if (__flag_ && (__is >> __saved_ln_ >> __saved_u2_))
            {
                __d.saved_ln_ = __saved_ln_;
                __d.saved_u2_ = __saved_u2_;
            }
        }

        return __is;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Type of real distribution
    using uniform_result_type =
        ::std::conditional_t<size_of_type_ % 2, sycl::vec<scalar_type, size_of_type_ + 1>, result_type>;

    // Distribution parameters
    scalar_type mean_;
    scalar_type stddev_;
    bool flag_ = false;
    scalar_type saved_ln_;
    scalar_type saved_u2_;

    // Static asserts
    static_assert(::std::is_floating_point_v<scalar_type>,
                  "oneapi::dpl::normal_distribution. Error: unsupported data type");

    // Real distribution for the conversion
    uniform_real_distribution<uniform_result_type> uniform_real_distribution_;
    using uniform_real_distr_param_type = typename uniform_real_distribution<uniform_result_type>::param_type;

    // Callback function
    template <typename _Type = float>
    inline scalar_type
    callback()
    {
        return ((scalar_type*)(internal::gaussian_sp_table))[1];
    }

    template <>
    inline scalar_type
    callback<double>()
    {
        return ((scalar_type*)(internal::gaussian_dp_table))[1];
    }

    // Get 2 * pi function
    template <typename _Type = float>
    inline scalar_type
    pi2()
    {
        return ((scalar_type*)(internal::gaussian_sp_table))[0];
    }

    template <>
    inline scalar_type
    pi2<double>()
    {
        return ((scalar_type*)(internal::gaussian_dp_table))[0];
    }

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr != 0), result_type>
    generate(_Engine& __engine, const param_type __params)
    {
        return generate_vec<_Ndistr, _Engine>(__engine, __params);
    }

    // Specialization of the scalar generation
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr == 0), result_type>
    generate(_Engine& __engine, const param_type __params)
    {
        result_type __res;
        result_type __ln;

        if (!flag_)
        {
            result_type __u1 =
                uniform_real_distribution_(__engine, uniform_real_distr_param_type(scalar_type{0.0}, scalar_type{1.0}));
            result_type __u2 =
                uniform_real_distribution_(__engine, uniform_real_distr_param_type(scalar_type{0.0}, scalar_type{1.0}));

            __ln = (__u1 == scalar_type{0.0}) ? callback<scalar_type>() : sycl::log(__u1);

            __res = __params.mean() +
                    __params.stddev() * (sycl::sqrt(-scalar_type{2.0} * __ln) * sycl::sin(pi2<scalar_type>() * __u2));

            saved_ln_ = __ln;
            saved_u2_ = __u2;
        }
        else
        {
            __res = __params.mean() + __params.stddev() * (sycl::sqrt(-scalar_type{2.0} * saved_ln_) *
                                                           sycl::cos(pi2<scalar_type>() * saved_u2_));
        }

        flag_ = !flag_;

        return __res;
    }

    // Specialization of the vector generation with size = [1; 2; 3]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N <= 3), result_type>
    generate_vec(_Engine& __engine, const param_type __params)
    {
        return generate_n_elems<_Engine>(__engine, __params, __N);
    }

    // Specialization of the vector generation with size = [4; 8; 16]
    template <int __N, class _Engine>
    ::std::enable_if_t<(__N > 3), result_type>
    generate_vec(_Engine& __engine, const param_type __params)
    {
        uniform_result_type __u;
        scalar_type __mean = __params.mean(), __stddev = __params.stddev();
        result_type __res;

        constexpr unsigned int __vec_size = __N / 2;
        sycl::vec<scalar_type, __vec_size> __sin, __cos;
        sycl::vec<scalar_type, __vec_size> __u1_transformed;

        __u = uniform_real_distribution_(__engine, uniform_real_distr_param_type(scalar_type{0.0}, scalar_type{1.0}),
                                         __N);

        sycl::vec<scalar_type, __vec_size> __u1 = __u.even();
        sycl::vec<scalar_type, __vec_size> __u2 = __u.odd();

        // Calculate sycl::log with callback
        __u1_transformed = select(sycl::log(__u1), sycl::vec<scalar_type, __vec_size>{callback<scalar_type>()},
                                  sycl::isequal(__u1, sycl::vec<scalar_type, __vec_size>{scalar_type{0.0}}));

        // Get sincos
        __sin = sycl::sincos(pi2<scalar_type>() * __u2, &__cos);

        if (!flag_)
        {
            __u1_transformed = sycl::sqrt(scalar_type{-2.0} * __u1_transformed);
            __res.even() = __u1_transformed * __sin * __stddev + __mean;
            __res.odd() = __u1_transformed * __cos * __stddev + __mean;

            // Flag is still false as code-branch for 4/8/16 vector sizes
        }
        else
        {
            __res[0] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * saved_ln_) *
                                            sycl::cos(pi2<scalar_type>() * saved_u2_));

            for (int __i = 1, __j = 0; __i < __N - 1; __i += 2, ++__j)
            {
                __res[__i] = (sycl::sqrt(scalar_type{-2.0} * __u1_transformed[__j]) * __sin[__j]) * __stddev + __mean;
                __res[__i + 1] =
                    (sycl::sqrt(scalar_type{-2.0} * __u1_transformed[__j]) * __cos[__j]) * __stddev + __mean;
            }

            __res[__N - 1] =
                (sycl::sqrt(scalar_type{-2.0} * __u1_transformed[__vec_size - 1]) * __sin[__vec_size - 1]) * __stddev +
                __mean;

            saved_ln_ = __u1_transformed[__vec_size - 1];
            saved_u2_ = __u2[__vec_size - 1];

            // Flag is still true as code-branch for 4/8/16 vector sizes
        }
        return __res;
    }

    // Implementation for the N vector's elements generation
    template <class _Engine>
    result_type
    generate_n_elems(_Engine& __engine, const param_type __params, unsigned int __N)
    {

        uniform_result_type __u;
        scalar_type __u1, __u2, __ln;
        scalar_type __sin, __cos;
        scalar_type __mean = __params.mean(), __stddev = __params.stddev();
        result_type __res;

        if (!flag_)
        {
            unsigned int __tail = __N % 2;
            __u = uniform_real_distribution_(
                __engine, uniform_real_distr_param_type(scalar_type{0.0}, scalar_type{1.0}), __N + __tail);

            for (unsigned int __i = 0; __i < __N - __tail; __i += 2)
            {
                __u1 = __u[__i];
                __u2 = __u[__i + 1];

                __sin = sycl::sincos(pi2<scalar_type>() * __u2, &__cos);

                __ln = (__u1 == scalar_type{0.0}) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__i] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * __sin);
                __res[__i + 1] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * __cos);
            }
            if (__tail)
            {
                __u1 = __u[__N - 1];
                __u2 = __u[__N];
                __ln = (__u1 == scalar_type{0.0}) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__N - 1] =
                    __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * sycl::sin(pi2<scalar_type>() * __u2));

                saved_ln_ = __ln;
                saved_u2_ = __u2;
                flag_ = true;
            }
        }
        else
        {
            __res[0] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * saved_ln_) *
                                            sycl::cos(pi2<scalar_type>() * saved_u2_));

            flag_ = false;

            unsigned int __tail = (__N - 1u) % 2u;

            __u = uniform_real_distribution_(
                __engine, uniform_real_distr_param_type(scalar_type{0.0}, scalar_type{1.0}), __N - 1u + __tail);

            for (unsigned int __i = 1; __i < (__N - __tail); __i += 2)
            {
                __u1 = __u[__i - 1];
                __u2 = __u[__i];

                __sin = sycl::sincos(pi2<scalar_type>() * __u2, &__cos);

                __ln = (__u1 == scalar_type{0.0}) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__i] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * __sin);
                __res[__i + 1] = __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * __cos);
            }
            if (__tail)
            {
                __u1 = __u[__N - 2];
                __u2 = __u[__N - 1];
                __ln = (__u1 == scalar_type{0.0}) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__N - 1] =
                    __mean + __stddev * (sycl::sqrt(-scalar_type{2.0} * __ln) * sycl::sin(pi2<scalar_type>() * __u2));

                saved_ln_ = __ln;
                saved_u2_ = __u2;
                flag_ = true;
            }
        }
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, class _Engine>
    ::std::enable_if_t<(_Ndistr != 0), result_type>
    result_portion_internal(_Engine& __engine, const param_type __params, unsigned int __N)
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

#endif // _ONEDPL_NORMAL_DISTRIBUTION_H
