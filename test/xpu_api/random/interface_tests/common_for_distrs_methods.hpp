// -*- C++ -*-
//===----------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//

#ifndef _ONEDPL_RANDOM_INTERFACE_TESTS_DISTRIBUTIONS_COMMON_H
#define _ONEDPL_RANDOM_INTERFACE_TESTS_DISTRIBUTIONS_COMMON_H

#include <oneapi/dpl/random>
static_assert(ONEDPL_HAS_RANDOM_NUMBERS >= 202409L);

#include <iostream>
#include <cmath>

constexpr auto SEED = 777;
constexpr auto N_GEN = 960;

template <typename T>
using Element_type = typename oneapi::dpl::internal::type_traits_t<T>::element_type;

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_int_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0};
    Element_type<T> b = std::numeric_limits<Element_type<T>>::max();
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().a() != a) || (distr.param().b() != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_real_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0.0};
    Element_type<T> b = Element_type<T>{1.0};
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().a() != a) || (distr.param().b() != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::normal_distribution<T>& distr)
{
    Element_type<T> mean = Element_type<T>{0.0};
    Element_type<T> stddev = Element_type<T>{1.0};
    return ((distr.mean() != mean) || (distr.stddev() != stddev) ||
            (distr.min() > -std::numeric_limits<Element_type<T>>::max()) ||
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || (distr.param().mean() != mean) ||
            (distr.param().stddev() != stddev));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::exponential_distribution<T>& distr)
{
    Element_type<T> lambda = Element_type<T>{1.0};
    return ((distr.lambda() != lambda) || (distr.min() != 0) ||
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().lambda() != lambda));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::bernoulli_distribution<T>& distr)
{
    double p = 0.5;
    return ((distr.p() != p) || (distr.min() != false) ||
            (distr.max() != true) || (distr.param().p() != p));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::geometric_distribution<T>& distr)
{
    double p = 0.5;
    return ((distr.p() != p) || (distr.min() != 0) ||
            (distr.max()  < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().p() != p));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::weibull_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{1.0};
    Element_type<T> b = Element_type<T>{1.0};
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != 0) || 
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) ||
            (distr.param().a() != a) || (distr.param().b() != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::lognormal_distribution<T>& distr)
{
    Element_type<T> m = Element_type<T>{0.0};
    Element_type<T> s = Element_type<T>{1.0};
    return ((distr.m() != m) || (distr.s() != s) ||
            (distr.min() != 0) || (distr.max() < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().m() != m) || (distr.param().s() != s));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::cauchy_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0.0};
    Element_type<T> b = Element_type<T>{1.0};
    return ((distr.a() != a) || (distr.b() != b) ||
            (distr.min() > std::numeric_limits<Element_type<T>>::lowest()) || 
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().a() != a) || (distr.param().b() != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::extreme_value_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0.0};
    Element_type<T> b = Element_type<T>{1.0};
    return ((distr.a() != a) || (distr.b() != b) ||
            (distr.min() > std::numeric_limits<Element_type<T>>::lowest()) || 
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().a() != a) || (distr.param().b() != b));
}


template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::uniform_int_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{0, 10};
    params2 = typename Distr::param_type{2, 8};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::uniform_real_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.0};
    params2 = typename Distr::param_type{-2.1, 2.2};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::exponential_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5};
    params2 = typename Distr::param_type{3.0};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::bernoulli_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{0.5};
    params2 = typename Distr::param_type{0.1};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::geometric_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{0.5};
    params2 = typename Distr::param_type{0.1};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::weibull_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.0};
    params2 = typename Distr::param_type{2.0, 40};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::lognormal_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.5};
    params2 = typename Distr::param_type{-2, 10};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::normal_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.5};
    params2 = typename Distr::param_type{-2, 10};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::cauchy_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.5};
    params2 = typename Distr::param_type{-2, 10};
}

template <typename Distr>
::std::enable_if_t<::std::is_same_v<Distr, oneapi::dpl::extreme_value_distribution<typename Distr::result_type>>>
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::param_type{1.5, 3.5};
    params2 = typename Distr::param_type{-2, 10};
}

template <class Distr>
std::int32_t
check_input_output(Distr& distr)
{
    using params_type = typename Distr::scalar_type;
    using result_type = typename Distr::result_type;

    std::int32_t status = 0;

    if constexpr (std::is_same_v<Distr, oneapi::dpl::bernoulli_distribution<result_type>>
            || std::is_same_v<Distr, oneapi::dpl::geometric_distribution<result_type>>) {
        double p = 0.5;

        std::stringstream s;
        s << p;

        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }
    else if constexpr (std::is_same_v<Distr, oneapi::dpl::exponential_distribution<result_type>>) {
        params_type lambda{1.0};

        std::stringstream s;
        s << lambda;

        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }
    else if constexpr (std::is_same_v<Distr, oneapi::dpl::normal_distribution<result_type>> 
                    || std::is_same_v<Distr, oneapi::dpl::lognormal_distribution<result_type>>) {
        params_type mean{0.0};
        params_type stddev{1.0};

        std::stringstream s;
        s << mean << ' ' << stddev;

        bool flag_ = true;
        params_type saved_ln_{1.2};
        params_type saved_u2_{1.3};

        s << ' ' << flag_ << ' ' << saved_ln_ << ' ' << saved_u2_;

        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }
    else if constexpr (std::is_same_v<Distr, oneapi::dpl::weibull_distribution<result_type>>) {
        params_type a{1.0};
        params_type b{1.0};

        std::stringstream s;
        s << a << ' ' << b;
        
        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }
    else if constexpr (std::is_same_v<Distr, oneapi::dpl::uniform_int_distribution<result_type>>) {
        params_type a{0};
        params_type b = std::numeric_limits<params_type>::max();

        std::stringstream s;
        s << a << ' ' << b;

        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }
    else { // uniform_real_distribution, cauchy_distribution, extreme_value_distribution
        params_type a{0.0};
        params_type b{1.0};

        std::stringstream s;
        s << a << ' ' << b;

        std::ostringstream out;
        std::istringstream in(s.str());

        in >> distr;
        status += check_params(distr);

        out << distr;
        if (!(out.str() == s.str()))
        {
            status += 1;
        }
    }

    return status;
}

template <class Distr>
bool
test_vec(sycl::queue& queue)
{

    typename Distr::param_type params1;
    typename Distr::param_type params2;

    make_param<Distr>(params1, params2);

    int sum = 0;

    // Memory allocation
    typename Distr::scalar_type res[N_GEN];
    constexpr std::int32_t num_elems =
        oneapi::dpl::internal::type_traits_t<typename Distr::result_type>::num_elems == 0
            ? 1
            : oneapi::dpl::internal::type_traits_t<typename Distr::result_type>::num_elems;

    // Random number generation
    {
        sycl::buffer<typename Distr::scalar_type> buffer(res, N_GEN);

        try
        {

            queue.submit([&](sycl::handler& cgh) {
                sycl::accessor acc(buffer, cgh, sycl::write_only);

                cgh.parallel_for<>(sycl::range<1>(N_GEN / (2 * num_elems)), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id() * num_elems;
                    oneapi::dpl::minstd_rand engine(SEED, offset);
                    Distr d1;
                    d1.param(params1);
                    Distr d2(params2);
                    d2.reset();
                    typename Distr::result_type res0 = d1(engine, params2, 1);
                    typename Distr::result_type res1 = d1(engine, params1, 1);
                    for (int i = 0; i < num_elems; ++i)
                    {
                        acc[offset * 2 + i] = res0[i];
                        acc[offset * 2 + num_elems + i] = res1[i];
                    }
                });
            });
        }
        catch (sycl::exception const& e)
        {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl;
            return 1;
        }

        queue.wait_and_throw();
        Distr distr;
        sum += check_params(distr);
    }

    return sum;
}

template <class Distr>
bool
test(sycl::queue& queue)
{

    typename Distr::param_type params1;
    typename Distr::param_type params2;

    make_param<Distr>(params1, params2);

    int status = 0;

    // Memory allocation
    typename Distr::scalar_type res[N_GEN];

    // Random number generation
    {
        {
            Distr _d1(2);
            Distr _d2(2);
            if (_d1 != _d2)
            {
                status += 1;
                std::cout << "Error: d1 != d2" << std::endl;
            }

            status += check_input_output(_d1);

            if (_d1 == _d2)
            {
                status += 1;
                std::cout << "Error: d1 == d2" << std::endl;
            }
        }

        sycl::buffer<typename Distr::scalar_type> buffer(res, N_GEN);

        try
        {
            queue.submit([&](sycl::handler& cgh) {
                sycl::stream out(1024, 256, cgh);
                cgh.single_task<>([=]() {
                    Distr distr;
                    out << "params: " << distr << sycl::endl;
                });
            });
            queue.wait_and_throw();

            queue.submit([&](sycl::handler& cgh) {
                sycl::accessor acc(buffer, cgh, sycl::write_only);

                cgh.parallel_for<>(sycl::range<1>(N_GEN / 2), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id();
                    oneapi::dpl::minstd_rand engine(SEED, offset);
                    Distr d1;
                    d1.param(params1);
                    Distr d2(params2);
                    d2.reset();
                    typename Distr::scalar_type res0 = d1(engine, params2);
                    typename Distr::scalar_type res1 = d1(engine, params1);
                    acc[offset * 2] = res0;
                    acc[offset * 2 + 1] = res1;
                });
            });
            queue.wait_and_throw();
        }
        catch (sycl::exception const& e)
        {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl;
            return 1;
        }

        Distr distr;
        status += check_params(distr);
    }

    return status;
}

#endif // _ONEDPL_RANDOM_INTERFACE_TESTS_DISTRIBUTIONS_COMMON_H
