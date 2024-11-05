// -*- C++ -*-
//===-- engines_methods.pass.cpp ------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Testing of different engines' methods

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#    include <oneapi/dpl/random>
static_assert(ONEDPL_HAS_RANDOM_NUMBERS >= 202409L);

#    include <iostream>
#    include <vector>

namespace ex = oneapi::dpl::experimental;

constexpr auto SEED = 777;
constexpr auto N_GEN = 960;
constexpr auto MINSTD_A = 48271;
constexpr auto MINSTD_C = 0;
constexpr auto MINSTD_M = 2147483647;
constexpr auto MINSTD_MIN = 1;
constexpr auto MINSTD_MAX = 2147483646;
constexpr auto RANLUX24_BASE_W = 24;
constexpr auto RANLUX24_BASE_S = 10;
constexpr auto RANLUX24_BASE_R = 24;
constexpr auto RANLUX24_BASE_MIN = 0;
constexpr auto RANLUX24_BASE_MAX = 16777215;
constexpr auto RANLUX24_P = 223;
constexpr auto RANLUX24_R = 23;
constexpr auto PHILOX4x32_W   = 32;
constexpr auto PHILOX4x64_W   = 64;
constexpr auto PHILOX4x32_N   = 4;
constexpr auto PHILOX4x64_N   = 4;
constexpr auto PHILOX4x32_R   = 10;
constexpr auto PHILOX4x64_R   = 10;
constexpr auto PHILOX4x32_MIN = 0;
constexpr auto PHILOX4x32_MAX = 0xFFFFFFFF;
constexpr auto PHILOX4x64_MIN = 0;
constexpr auto PHILOX4x64_MAX = 0xFFFFFFFFFFFFFFFF;
constexpr auto PHILOX_SEED = 20111115u;

std::int32_t
check_params(oneapi::dpl::minstd_rand& engine)
{
    return ((oneapi::dpl::minstd_rand::multiplier != MINSTD_A) || (oneapi::dpl::minstd_rand::increment != MINSTD_C) ||
            (oneapi::dpl::minstd_rand::modulus != MINSTD_M) || (engine.min() != MINSTD_MIN) ||
            (engine.max() != MINSTD_MAX));
}

template <int N>
std::int32_t
check_params(oneapi::dpl::minstd_rand_vec<N>& engine)
{
    return ((oneapi::dpl::minstd_rand_vec<N>::multiplier != MINSTD_A) ||
            (oneapi::dpl::minstd_rand_vec<N>::increment != MINSTD_C) ||
            (oneapi::dpl::minstd_rand_vec<N>::modulus != MINSTD_M) || (engine.min() != MINSTD_MIN) ||
            (engine.max() != MINSTD_MAX));
}

std::int32_t
check_params(oneapi::dpl::ranlux24_base& engine)
{
    return ((oneapi::dpl::ranlux24_base::word_size != RANLUX24_BASE_W) ||
            (oneapi::dpl::ranlux24_base::short_lag != RANLUX24_BASE_S) ||
            (oneapi::dpl::ranlux24_base::long_lag != RANLUX24_BASE_R) || (engine.min() != RANLUX24_BASE_MIN) ||
            (engine.max() != RANLUX24_BASE_MAX));
}

template <int N>
std::int32_t
check_params(oneapi::dpl::ranlux24_base_vec<N>& engine)
{
    return ((oneapi::dpl::ranlux24_base_vec<N>::word_size != RANLUX24_BASE_W) ||
            (oneapi::dpl::ranlux24_base_vec<N>::short_lag != RANLUX24_BASE_S) ||
            (oneapi::dpl::ranlux24_base_vec<N>::long_lag != RANLUX24_BASE_R) || (engine.min() != RANLUX24_BASE_MIN) ||
            (engine.max() != RANLUX24_BASE_MAX));
}

std::int32_t
check_params(oneapi::dpl::ranlux24& engine)
{
    return ((oneapi::dpl::ranlux24::block_size != RANLUX24_P) || (oneapi::dpl::ranlux24::used_block != RANLUX24_R) ||
            (engine.min() != RANLUX24_BASE_MIN) || (engine.max() != RANLUX24_BASE_MAX));
}

template <int N>
std::int32_t
check_params(oneapi::dpl::ranlux24_vec<N>& engine)
{
    return ((oneapi::dpl::ranlux24_vec<N>::block_size != RANLUX24_P) ||
            (oneapi::dpl::ranlux24_vec<N>::used_block != RANLUX24_R) || (engine.min() != RANLUX24_BASE_MIN) ||
            (engine.max() != RANLUX24_BASE_MAX));
}

std::int32_t
check_params(ex::philox4x32& engine)
{
    return ((ex::philox4x32::word_size    != PHILOX4x32_W) || 
            (ex::philox4x32::word_count   != PHILOX4x32_N) ||
            (ex::philox4x32::round_count  != PHILOX4x32_R) ||
            (ex::philox4x32::default_seed != PHILOX_SEED)  ||
            (engine.min() != PHILOX4x32_MIN) || (engine.max() != PHILOX4x32_MAX));
}
template <int N>
std::int32_t
check_params(ex::philox4x32_vec<N>& engine)
{
    return ((ex::philox4x32_vec<N>::word_size    != PHILOX4x32_W) || 
            (ex::philox4x32_vec<N>::word_count   != PHILOX4x32_N) ||
            (ex::philox4x32_vec<N>::round_count  != PHILOX4x32_R) ||
            (ex::philox4x32_vec<N>::default_seed != PHILOX_SEED)  ||
            (engine.min() != PHILOX4x32_MIN) || (engine.max() != PHILOX4x32_MAX));
}

std::int32_t
check_params(ex::philox4x64& engine)
{
    return ((ex::philox4x64::word_size    != PHILOX4x64_W) || 
            (ex::philox4x64::word_count   != PHILOX4x64_N) ||
            (ex::philox4x64::round_count  != PHILOX4x64_R) ||
            (ex::philox4x64::default_seed != PHILOX_SEED)  ||
            (engine.min() != PHILOX4x64_MIN) || (engine.max() != PHILOX4x64_MAX));
}
template <int N>
std::int32_t
check_params(ex::philox4x64_vec<N>& engine)
{
    return ((ex::philox4x64_vec<N>::word_size    != PHILOX4x64_W) || 
            (ex::philox4x64_vec<N>::word_count   != PHILOX4x64_N) ||
            (ex::philox4x64_vec<N>::round_count  != PHILOX4x64_R) ||
            (ex::philox4x64_vec<N>::default_seed != PHILOX_SEED)  ||
            (engine.min() != PHILOX4x64_MIN) || (engine.max() != PHILOX4x64_MAX));
}

template <class Engine>
class
test_vec
{
public:
    bool run(sycl::queue& queue)
    {
        using result_type = typename Engine::scalar_type;

        int sum = 0;

        // Memory allocation
        std::vector<std::int32_t> dpstd_res(N_GEN);
        constexpr std::int32_t num_elems =
            oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems == 0
                ? 1
                : oneapi::dpl::internal::type_traits_t<result_type>::num_elems;

        // Random number generation
        {
            sycl::buffer<std::int32_t> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

            try
            {
                queue.submit([&](sycl::handler& cgh) {
                    auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                    cgh.parallel_for<>(sycl::range<1>(N_GEN), [=](sycl::item<1> idx) {
                        unsigned long long offset = idx.get_linear_id();
                        Engine engine0(SEED);
                        Engine engine1;
                        engine1.seed(SEED);
                        engine0.discard(offset);
                        engine1.discard(offset);
                        typename Engine::result_type res0;
                        res0 = engine0();
                        typename Engine::result_type res1 = engine1();
                        std::int32_t is_inequal = 0;
                        for (std::int32_t i = 0; i < num_elems; ++i)
                        {
                            if (res0[i] != res1[i])
                            {
                                is_inequal = 1;
                            }
                        }
                        dpstd_acc[offset] = is_inequal;
                    });
                });
                auto dpstd_acc = dpstd_buffer.get_host_access(sycl::write_only);
                for (int i = 0; i < N_GEN; ++i)
                {
                    sum += dpstd_acc[i];
                }
                if (sum)
                {
                    std::cout << "Error occurred in " << sum << " elements" << std::endl;
                }
            }
            catch (sycl::exception const& e)
            {
                std::cout << "\t\tSYCL exception during generation\n"
                          << e.what() << std::endl;
                return 1;
            }

            queue.wait_and_throw();
            Engine engine;
            sum += check_params(engine);
        }

        return sum;
    }
};

template <int N>
class
test_vec<oneapi::dpl::ranlux24_vec<N>>
{
public:
    bool run(sycl::queue& queue)
    {
        using result_type = typename oneapi::dpl::ranlux24_vec<N>::scalar_type;

        int sum = 0;

        // Memory allocation
        std::vector<std::int32_t> dpstd_res(N_GEN);
        constexpr std::int32_t num_elems =
            oneapi::dpl::internal::type_traits_t<typename oneapi::dpl::ranlux24_vec<N>::result_type>::num_elems == 0
                ? 1
                : oneapi::dpl::internal::type_traits_t<result_type>::num_elems;

        // Random number generation
        {
            sycl::buffer<std::int32_t> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

            try
            {
                queue.submit([&](sycl::handler& cgh) {
                    auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                    cgh.parallel_for<>(sycl::range<1>(N_GEN), [=](sycl::item<1> idx) {
                        unsigned long long offset = idx.get_linear_id();
                        oneapi::dpl::ranlux24_vec<N> engine0(SEED);
                        oneapi::dpl::ranlux24_vec<N> engine1;
                        engine1.seed(SEED);
                        engine0.discard(offset);
                        engine1.discard(offset);
                        typename oneapi::dpl::ranlux24_vec<N>::result_type res0;
                        oneapi::dpl::ranlux24_vec<N> engine(engine1);
                        auto eng = engine.base().min();
                        res0 = engine() + eng;
                        typename oneapi::dpl::ranlux24_vec<N>::result_type res1 = engine1();
                        std::int32_t is_inequal = 0;
                        for (std::int32_t i = 0; i < num_elems; ++i)
                        {
                            if (res0[i] != res1[i] + eng)
                            {
                                is_inequal = 1;
                            }
                        }
                        dpstd_acc[offset] = is_inequal;
                    });
                });
                auto dpstd_acc = dpstd_buffer.get_host_access(sycl::write_only);
                for (int i = 0; i < N_GEN; ++i)
                {
                    sum += dpstd_acc[i];
                }
                if (sum)
                {
                    std::cout << "Error occurred in " << sum << " elements" << std::endl;
                }
                queue.wait_and_throw();
            }
            catch (sycl::exception const& e)
            {
                std::cout << "\t\tSYCL exception during generation\n"
                          << e.what() << std::endl;
                return 1;
            }
            oneapi::dpl::ranlux24_vec<N> engine;
            sum += check_params(engine);
        }

        return sum;
    }
};

template <class Engine>
class
test
{
public:
    bool run(sycl::queue& queue)
    {
        using result_type = typename Engine::scalar_type;

        int sum = 0;

        // Memory allocation
        std::vector<std::int32_t> dpstd_res(N_GEN);

        // Random number generation
        {
            {
                Engine e1;
                e1.discard(100);

                std::ostringstream os;
                os << e1;

                Engine e2;

                std::istringstream in(os.str());
                in >> e2;

                if (e1 != e2)
                    sum += 1;
            }

            sycl::buffer<std::int32_t> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

            try
            {
                queue.submit([&](sycl::handler& cgh) {
                    sycl::stream out(1024, 512, cgh);
                    cgh.single_task<>([=]() {
                        Engine engine(SEED);
                        out << "state: " << engine << sycl::endl;
                    });
                });
                queue.wait_and_throw();

                queue.submit([&](sycl::handler& cgh) {
                    auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<>(sycl::range<1>(N_GEN), [=](sycl::item<1> idx) {
                        unsigned long long offset = idx.get_linear_id() + 1;
                        Engine engine0(SEED);
                        Engine engine1;
                        engine1.seed(SEED);
                        engine0.discard(offset);
                        if (engine0 == engine1)
                        {
                            dpstd_acc[offset] = 1;
                        }
                        else
                        {
                            dpstd_acc[offset] = 0;
                        }
                        engine1.discard(offset);
                        if (engine0 != engine1)
                        {
                            dpstd_acc[offset] += 1;
                        }
                        result_type res0;
                        res0 = engine0();
                        result_type res1 = engine1();
                        if (res0 != res1)
                        {
                            dpstd_acc[offset] += 1;
                        }
                    });
                });
                auto dpstd_acc = dpstd_buffer.get_host_access(sycl::write_only);
                for (int i = 0; i < N_GEN; ++i)
                {
                    sum += dpstd_acc[i];
                }
                if (sum)
                {
                    std::cout << "Error occurred in " << sum << " elements" << std::endl;
                }
            }
            catch (sycl::exception const& e)
            {
                std::cout << "\t\tSYCL exception during generation\n"
                          << e.what() << std::endl;
                return 1;
            }

            queue.wait_and_throw();
            Engine engine;
            sum += check_params(engine);
        }

        return sum;
    }
};

template <>
class
test<oneapi::dpl::ranlux24>
{
public:
    bool run(sycl::queue& queue)
    {
        using result_type = typename oneapi::dpl::ranlux24::scalar_type;

        int sum = 0;

        // Memory allocation
        std::vector<std::int32_t> dpstd_res(N_GEN);

        // Random number generation
        {
            {
                oneapi::dpl::ranlux24 e1;
                e1.discard(100);

                std::ostringstream os;
                os << e1;

                oneapi::dpl::ranlux24 e2;

                std::istringstream in(os.str());
                in >> e2;

                if (e1 != e2)
                    sum += 1;
            }

            sycl::buffer<std::int32_t> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

            try
            {
                queue.submit([&](sycl::handler& cgh) {
                    sycl::stream out(1024, 512, cgh);
                    cgh.single_task<>([=]() {
                        oneapi::dpl::ranlux24 engine(SEED);
                        out << "state: " << engine << sycl::endl;
                    });
                });
                queue.wait_and_throw();

                queue.submit([&](sycl::handler& cgh) {
                    auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<>(sycl::range<1>(N_GEN), [=](sycl::item<1> idx) {
                        unsigned long long offset = idx.get_linear_id() + 1;
                        oneapi::dpl::ranlux24 engine0(SEED);
                        oneapi::dpl::ranlux24 engine1;
                        engine1.seed(SEED);
                        engine0.discard(offset);
                        if (engine0 == engine1)
                        {
                            dpstd_acc[offset] = 1;
                        }
                        else
                        {
                            dpstd_acc[offset] = 0;
                        }
                        engine1.discard(offset);
                        if (engine0 != engine1)
                        {
                            dpstd_acc[offset] += 1;
                        }
                        result_type res0;
                        oneapi::dpl::ranlux24 engine(engine1);
                        auto eng = engine.base().min();
                        res0 = engine() + eng;
                        result_type res1 = engine1();
                        if (res0 != res1 + eng)
                        {
                            dpstd_acc[offset] += 1;
                        }
                    });
                });
                auto dpstd_acc = dpstd_buffer.get_host_access(sycl::write_only);
                for (int i = 0; i < N_GEN; ++i)
                {
                    sum += dpstd_acc[i];
                }
                if (sum)
                {
                    std::cout << "Error occurred in " << sum << " elements" << std::endl;
                }
            }
            catch (sycl::exception const& e)
            {
                std::cout << "\t\tSYCL exception during generation\n"
                          << e.what() << std::endl;
                return 1;
            }

            queue.wait_and_throw();
            oneapi::dpl::ranlux24 engine;
            sum += check_params(engine);
        }

        return sum;
    }
};

#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    std::int32_t err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "linear_congruential_engine<48271, 0, 2147483647>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::minstd_rand>{}.run(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::minstd_rand_vec<16>>{}.run(queue);
    err += test_vec<oneapi::dpl::minstd_rand_vec<8>>{}.run(queue);
    err += test_vec<oneapi::dpl::minstd_rand_vec<4>>{}.run(queue);
    err += test_vec<oneapi::dpl::minstd_rand_vec<3>>{}.run(queue);
    err += test_vec<oneapi::dpl::minstd_rand_vec<2>>{}.run(queue);
    err += test_vec<oneapi::dpl::minstd_rand_vec<1>>{}.run(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "subtract_with_carry_engine<24, 10, 24>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::ranlux24_base>{}.run(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::ranlux24_base_vec<16>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_base_vec<8>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_base_vec<4>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_base_vec<3>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_base_vec<2>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_base_vec<1>>{}.run(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base, 223, 23>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::ranlux24>{}.run(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::ranlux24_vec<16>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_vec<8>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_vec<4>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_vec<3>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_vec<2>>{}.run(queue);
    err += test_vec<oneapi::dpl::ranlux24_vec<1>>{}.run(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "philox_engine<uint_fast32_t, 32, 4, 10,\n"
                 "0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<ex::philox4x32>{}.run(queue);
    #if TEST_LONG_RUN
    err += test_vec<ex::philox4x32_vec<16>>{}.run(queue);
    err += test_vec<ex::philox4x32_vec<8>>{}.run(queue);
    err += test_vec<ex::philox4x32_vec<4>>{}.run(queue);
    err += test_vec<ex::philox4x32_vec<3>>{}.run(queue);
    err += test_vec<ex::philox4x32_vec<2>>{}.run(queue);
    err += test_vec<ex::philox4x32_vec<1>>{}.run(queue);
    #endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");
    
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "philox_engine<uint_fast64_t, 64, 4, 10,\n"
                 "0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<ex::philox4x64>{}.run(queue);
    #if TEST_LONG_RUN
    err += test_vec<ex::philox4x64_vec<16>>{}.run(queue);
    err += test_vec<ex::philox4x64_vec<8>>{}.run(queue);
    err += test_vec<ex::philox4x64_vec<4>>{}.run(queue);
    err += test_vec<ex::philox4x64_vec<3>>{}.run(queue);
    err += test_vec<ex::philox4x64_vec<2>>{}.run(queue);
    err += test_vec<ex::philox4x64_vec<1>>{}.run(queue);
    #endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
