// -*- C++ -*-
//===-- uniform_real_distribution.h ---------------------------------------===//
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
// Public header file provides implementation for Uniform Real Distribution

#ifndef _ONEDPL_UNIFORM_REAL_DISTRIBUTION_H
#define _ONEDPL_UNIFORM_REAL_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{

template <class _RealType = double>
class uniform_real_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;
    class param_type
    {
      public:
        using distribution_type = uniform_real_distribution<result_type>;
        param_type() : param_type(scalar_type{0.0}) {}
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
    uniform_real_distribution() : uniform_real_distribution(scalar_type{0.0}) {}
    explicit uniform_real_distribution(scalar_type __a, scalar_type __b = scalar_type{1.0}) : a_(__a), b_(__b) {}
    explicit uniform_real_distribution(const param_type& __params) : a_(__params.a()), b_(__params.b()) {}

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
        return a();
    }

    scalar_type
    max() const
    {
        return b();
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
        return generate<size_of_type_, internal::type_traits_t<typename _Engine::result_type>::num_elems, _Engine>(
            __engine, __params);
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
        return result_portion_internal<size_of_type_, internal::type_traits_t<typename _Engine::result_type>::num_elems,
                                       _Engine>(__engine, __params, __random_nums);
    }

    friend bool
    operator==(const uniform_real_distribution& __x, const uniform_real_distribution& __y)
    {
        return __x.param() == __y.param();
    }

    friend bool
    operator!=(const uniform_real_distribution& __x, const uniform_real_distribution& __y)
    {
        return !(__x == __y);
    }

    template <class CharT, class Traits>
    friend ::std::basic_ostream<CharT, Traits>&
    operator<<(::std::basic_ostream<CharT, Traits>& __os, const uniform_real_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__os);

        __os.setf(std::ios_base::dec | std::ios_base::left);
        CharT __sp = __os.widen(' ');
        __os.fill(__sp);

        return __os << __d.a() << __sp << __d.b();
    }

    friend const sycl::stream&
    operator<<(const sycl::stream& __os, const uniform_real_distribution& __d)
    {
        return __os << __d.a() << ' ' << __d.b();
    }

    template <class CharT, class Traits>
    friend ::std::basic_istream<CharT, Traits>&
    operator>>(::std::basic_istream<CharT, Traits>& __is, uniform_real_distribution& __d)
    {
        internal::save_stream_flags<CharT, Traits> __flags(__is);

        __is.setf(std::ios_base::dec);

        uniform_real_distribution::scalar_type __a;
        uniform_real_distribution::scalar_type __b;

        if (__is >> __a >> __b)
            __d.param(uniform_real_distribution::param_type(__a, __b));

        return __is;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Static asserts
    static_assert(::std::is_floating_point_v<scalar_type>,
                  "oneapi::dpl::uniform_real_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type a_;
    scalar_type b_;

    template <typename _IntegerT, typename _Engine>
    inline scalar_type
    make_real_uniform(_IntegerT __int_val, _Engine& __engine, const param_type& __params)
    {
        return static_cast<scalar_type>(
            ((__int_val - __engine.min()) /
             (static_cast<scalar_type>(1) + static_cast<scalar_type>(__engine.max() - __engine.min()))) *
                (__params.b() - __params.a()) +
            __params.a());
    }

    // Implementation for generate function
    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr == _Nengine) & (_Ndistr != 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        auto __engine_output = __engine();
        result_type __res{};

        for (int __i = 0; __i < _Ndistr; ++__i)
        {
            __res[__i] = make_real_uniform(__engine_output[__i], __engine, __params);
        }

        return __res;
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr == _Nengine) & (_Ndistr == 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        return make_real_uniform(__engine(), __engine, __params);
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr < _Nengine) & (_Ndistr != 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        auto __engine_output = __engine(_Ndistr);
        result_type __res{};
        for (int __i = 0; __i < _Ndistr; ++__i)
        {
            __res[__i] = make_real_uniform(__engine_output[__i], __engine, __params);
        }
        return __res;
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr < _Nengine) & (_Ndistr == 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        return make_real_uniform(__engine(1)[0], __engine, __params);
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr > _Nengine) & (_Nengine != 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        sycl::vec<scalar_type, _Ndistr> __res{};
        int __i;
        constexpr int __tail_size = _Ndistr % _Nengine;
        for (__i = 0; __i < _Ndistr - __tail_size; __i += _Nengine)
        {
            auto __engine_output = __engine();
            for (int __j = 0; __j < _Nengine; ++__j)
            {
                __res[__i + __j] = make_real_uniform(__engine_output[__j], __engine, __params);
            }
        }

        if (__tail_size)
        {
            __i = _Ndistr - __tail_size;
            auto __engine_output = __engine(__tail_size);
            for (int __j = 0; __j < __tail_size; __j++)
            {
                __res[__i + __j] = make_real_uniform(__engine_output[__j], __engine, __params);
            }
        }
        return __res;
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr > _Nengine) & (_Nengine == 0)), result_type>
    generate(_Engine& __engine, const param_type& __params)
    {
        sycl::vec<scalar_type, _Ndistr> __res{};
        for (int __i = 0; __i < _Ndistr; ++__i)
        {
            __res[__i] = make_real_uniform(__engine(), __engine, __params);
        }
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr <= _Nengine) & (_Ndistr != 0)), result_type>
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        auto __engine_output = __engine(__N);
        result_type __res{};
        for (unsigned int __i = 0; __i < __N; ++__i)
        {
            __res[__i] = make_real_uniform(__engine_output[__i], __engine, __params);
        }
        return __res;
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr > _Nengine) & (_Nengine != 0)), result_type>
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res{};
        unsigned int __i;

        if (_Nengine >= __N)
        {
            auto __engine_output = __engine(__N);
            for (__i = 0; __i < __N; ++__i)
            {
                __res[__i] = make_real_uniform(__engine_output[__i], __engine, __params);
            }
        }
        else
        {
            unsigned int __tail_size = __N % _Nengine;
            for (__i = 0; __i < __N; __i += _Nengine)
            {
                auto __engine_output = __engine();
                for (int __j = 0; __j < _Nengine; ++__j)
                {
                    __res[__i + __j] = make_real_uniform(__engine_output[__j], __engine, __params);
                }
            }
            if (__tail_size)
            {
                __i = _Ndistr - __tail_size;
                auto __engine_output = __engine(__tail_size);

                for (unsigned int __j = 0; __j < __tail_size; ++__j)
                {
                    __res[__i + __j] = make_real_uniform(__engine_output[__j], __engine, __params);
                }
            }
        }
        return __res;
    }

    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<((_Ndistr > _Nengine) & (_Nengine == 0)), result_type>
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res{};
        for (unsigned int __i = 0; __i < __N; ++__i)
        {
            __res[__i] = make_real_uniform(__engine(), __engine, __params);
        }
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, int _Nengine, class _Engine>
    ::std::enable_if_t<(_Ndistr != 0), result_type>
    result_portion_internal(_Engine& __engine, const param_type __params, unsigned int __N)
    {
        result_type __part_vec;
        if (__N == 0)
            return __part_vec;
        else if (__N >= _Ndistr)
            return operator()(__engine, __params);

        __part_vec = generate_n_elems<_Ndistr, _Nengine, _Engine>(__engine, __params, __N);
        return __part_vec;
    }
};

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UNIFORM_REAL_DISTRIBUTION_H
