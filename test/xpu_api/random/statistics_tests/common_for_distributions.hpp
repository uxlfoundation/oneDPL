// -*- C++ -*-
//===----------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//

#ifndef _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_DISTRS_H
#define _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_DISTRS_H

#include <iostream>
#include <random>
#include <limits>
#include <oneapi/dpl/random>
#include <math.h>

#include "statistics_common.h"

// Engine parameters
constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;
constexpr auto seed = 777;

template <typename T>
using Element_type = typename oneapi::dpl::internal::type_traits_t<T>::element_type;

template <typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::exponential_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType lambda)
{
    // theoretical moments
    double tM = 1 / lambda;
    double tD = 1 / (lambda * lambda);
    double tQ = 9 / (lambda * lambda * lambda * lambda);

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::extreme_value_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType _a, ScalarRealType _b)
{
    // theoretical moments
    const double y = 0.5772156649015328606065120;
    const double pi = 3.1415926535897932384626433;
    double tM = _a + _b * y;
    double tD = pi * pi / 6.0 * _b * _b;
    double tQ = 27.0 / 5.0 * tD * tD;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <typename ScalarIntType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::geometric_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarIntType>& samples, double p)
{
    // theoretical moments
    double tM = (1 - p) / p;
    double tD = (1 - p) / (p * p);
    double tQ = (9 - 9 * p + p * p) * (1 - p) / (p * p * p * p);

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template<typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::lognormal_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType mean, ScalarRealType stddev) {
    // theoretical moments
    double tM = exp(mean + stddev * stddev / 2);
    double tD = (exp(stddev * stddev) - 1) * exp(2 * mean + stddev * stddev);
    double tQ = (exp(4 * stddev * stddev) + 2 * exp(3 * stddev * stddev) + 3 * exp(2 * stddev * stddev) - 3) * tD * tD;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template<typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::normal_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType mean, ScalarRealType stddev) {
    // theoretical moments
    double tM = mean;
    double tD = stddev * stddev;
    double tQ = 720.0 * stddev * stddev * stddev * stddev;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template<typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::uniform_real_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType left, ScalarRealType right)
{
    // theoretical moments
    double tM = (right + left) / 2.0;
    double tD = ((right - left) * (right - left)) / 12.0;
    double tQ = ((right - left) * (right - left) * (right - left) * (right - left)) / 80.0;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <typename ScalarRealType, typename Distr>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::weibull_distribution<typename Distr::result_type>>, int>
statistics_check(int nsamples, const std::vector<ScalarRealType>& samples, ScalarRealType _a, ScalarRealType _b)
{
    // theoretical moments
    double G1 = sycl::tgamma(1 + 1 / _a);
    double G2 = sycl::tgamma(1 + 2 / _a);
    double G3 = sycl::tgamma(1 + 3 / _a);
    double G4 = sycl::tgamma(1 + 4 / _a);
    double tM = _b * G1;
    double tD = _b * _b * (G2 - G1 * G1);
    double tQ = _b * _b * _b * _b * ((-3) * G1 * G1 * G1 * G1 + 12 * G1 * G1 * G2 - 4 * G1 * G3 + G4 - 6 * G2 * G1 *G1);

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <class Distr, class UIntType, class Engine = oneapi::dpl::linear_congruential_engine<UIntType, a, c, m>, class... Args>
int
test(sycl::queue& queue, int nsamples, Args... params)
{
    using Type = typename Distr::result_type;

    // memory allocation
    std::vector<Element_type<Type>> samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<Type>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<Type>::num_elems;

    // generation
    {
        sycl::buffer<Element_type<Type>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            auto acc = buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                Engine engine(seed);
                engine.discard(offset);
                Distr distr(params...);

                sycl::vec<Element_type<Type>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc);
            });
        });
    }

    // statistics check
    int err = statistics_check<Element_type<Type>, Distr>(nsamples, samples, params...);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <class Distr, class UIntType, class Engine = oneapi::dpl::linear_congruential_engine<UIntType, a, c, m>,  class... Args>
int
test_portion(sycl::queue& queue, int nsamples, unsigned int part, Args... params)
{
    using Type = typename Distr::result_type;

    // memory allocation
    std::vector<Element_type<Type>> samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<Type>::num_elems == 0
                                           ? 1
                                           : oneapi::dpl::internal::type_traits_t<Type>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<Element_type<Type>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            auto acc = buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * n_elems;
                Engine engine(seed);
                engine.discard(offset);
                Distr distr(params...);

                sycl::vec<Element_type<Type>, num_elems> res = distr(engine, part);
                for (int i = 0; i < n_elems; ++i)
                    acc[offset + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // statistics check
    int err = statistics_check<Element_type<Type>, Distr>(nsamples, samples, params...);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::exponential_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> lambda_array[nparams] = {0.5, 1.5};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "exponential_distribution test<type>, lambda = " << lambda_array[i]
                  << ", nsamples  = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, lambda_array[i])) {
            return 1;
        }
    }
    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::extreme_value_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> a_array [nparams] = {2.0, -10.0};
    oneapi::dpl::internal::element_type_t<real_type> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "extreme_value_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples  = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, a_array[i], b_array[i])) {
            return 1;
        }
    }
    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::geometric_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    constexpr int nparams = 2;

    double p_array[nparams] = {0.2, 0.9};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "geometric_distribution test<type>, p = " << p_array[i]
                  << ", nsamples  = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, p_array[i])) {
            return 1;
        }
    }
    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::lognormal_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<real_type> stddev_array [nparams] = {1.0, 1000.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "lognormal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, mean_array[i], stddev_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::normal_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<real_type> stddev_array [nparams] = {1.0, 1000.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "normal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, mean_array[i], stddev_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType, class Engine = oneapi::dpl::linear_congruential_engine<UIntType, a, c, m>>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::uniform_real_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> left_array [nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<real_type> right_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples  = " << nsamples;
        if (test<Distr, UIntType, Engine>(queue, nsamples, left_array[i], right_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::weibull_distribution<typename Distr::result_type>>, int>
tests_set(sycl::queue& queue, int nsamples)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> a_array [nparams] = {2.0, 10.0};
    oneapi::dpl::internal::element_type_t<real_type> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "weibull_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples  = " << nsamples;
        if (test<Distr, UIntType>(queue, nsamples, a_array[i], b_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::exponential_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> lambda_array[nparams] = {0.5, 1.5};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "exponential_distribution test<type>, lambda = " << lambda_array[i] << ", nsamples = " << nsamples
                  << ", part = " << part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, lambda_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::extreme_value_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> a_array [nparams] = {2.0, -10.0};
    oneapi::dpl::internal::element_type_t<real_type> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "extreme_value_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, a_array[i], b_array[i])) {
            return 1;
        }
    }

    return 0;
}

template <class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::geometric_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    double p_array[nparams] = {0.2, 0.9};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "geometric_distribution test<type>, p = " << p_array[i] << ", nsamples = " << nsamples
                  << ", part = " << part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, p_array[i])) {
            return 1;
        }
    }

    return 0;
}

template<class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::lognormal_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<real_type> stddev_array [nparams] = {1.0, 1000.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "lognormal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples << ", part = "<< part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, mean_array[i], stddev_array[i])) {
            return 1;
        }
    }
    return 0;
}

template<class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::normal_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<real_type> stddev_array [nparams] = {1.0, 1000.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "normal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples << ", part = "<< part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, mean_array[i], stddev_array[i])) {
            return 1;
        }
    }
    return 0;
}

template<class Distr, class UIntType, class Engine = oneapi::dpl::linear_congruential_engine<UIntType, a, c, m>>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::uniform_real_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> left_array [nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<real_type> right_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        if (test_portion<Distr, UIntType, Engine>(queue, nsamples, part, left_array[i], right_array[i])) {
            return 1;
        }
    }
    return 0;
}

template<class Distr, class UIntType>
std::enable_if_t<std::is_same_v<Distr, oneapi::dpl::weibull_distribution<typename Distr::result_type>>, int>
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    using real_type = typename Distr::result_type;

    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<real_type> a_array [nparams] = {2.0, 10.0};
    oneapi::dpl::internal::element_type_t<real_type> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i) {
        std::cout << "weibull_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        if (test_portion<Distr, UIntType>(queue, nsamples, part, a_array[i], b_array[i])) {
            return 1;
        }
    }
    return 0;
}

#endif // _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_DISTRS_H
