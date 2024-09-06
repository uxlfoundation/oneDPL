// -*- C++ -*-
//===-- philox_unit_test.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Unit test for Philox random number generation engine - corner-case testing functions

#include "support/utils.h"
#include "support/test_config.h"

#include <random>
#include <oneapi/dpl/random>

namespace ex = oneapi::dpl::experimental;

/* Corner-case testing functions */

template <typename Engine>
int seed_test();

template <typename Engine>
int discard_test();

template <typename Engine>
int set_counter_conformance_test();

template <typename Engine>
int skip_test();

template <typename Engine>
int counter_overflow_test();

template <typename Engine>
int discard_overflow_test();

using philox4x32_w5 = ex::philox_engine<std::uint_fast32_t, 5, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w15 = ex::philox_engine<std::uint_fast32_t, 15, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w18 = ex::philox_engine<std::uint_fast32_t, 18, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w30 = ex::philox_engine<std::uint_fast32_t, 30, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x64_w5 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 5, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                             0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w15 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 15, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                             0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w18 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 18, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                             0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w25 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 25, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                             0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w49 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 49, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                             0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

int
main()
{
    int err = 0;

    /* Test of the Philox engine with pre-defined standard parameters */
    std::cout << "void seed_test() [Engine = philox4x32]";
    err += seed_test<ex::philox4x32>();
    std::cout << "void seed_test() [Engine = philox4x64]";
    err += seed_test<ex::philox4x64>();

    std::cout << "void discard_test() [Engine = philox4x32]";
    err += discard_test<ex::philox4x32>();
    std::cout << "void discard_test() [Engine = philox4x64]";
    err += discard_test<ex::philox4x64>();

    std::cout << "void set_counter_conformance_test() [Engine = philox4x32]";
    err += set_counter_conformance_test<ex::philox4x32>();
    std::cout << "void set_counter_conformance_test() [Engine = philox4x64]";
    err += set_counter_conformance_test<ex::philox4x64>();

    std::cout << "void skip_test() [Engine = philox4x32]";
    err += skip_test<ex::philox4x32>();
    std::cout << "void skip_test() [Engine = philox4x64]";
    err += skip_test<ex::philox4x64>();

    std::cout << "void counter_overflow_test() [Engine = philox4x32]";
    err += counter_overflow_test<ex::philox4x32>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64]";
    err += counter_overflow_test<ex::philox4x64>();

    std::cout << "void discard_overflow_test() [Engine = philox4x32]";
    err += discard_overflow_test<ex::philox4x32>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64]";
    err += discard_overflow_test<ex::philox4x64>();

    EXPECT_TRUE(!err, "Test FAILED");

    /* Test of the Philox engine with non-standard parameters */
    std::cout << "void counter_overflow_test() [Engine = philox4x32_w5]";
    err += counter_overflow_test<philox4x32_w5>();
    std::cout << "void counter_overflow_test() [Engine = philox4x32_w15]";
    err += counter_overflow_test<philox4x32_w15>();
    std::cout << "void counter_overflow_test() [Engine = philox4x32_w18]";
    err += counter_overflow_test<philox4x32_w18>();
    std::cout << "void counter_overflow_test() [Engine = philox4x32_w30]";
    err += counter_overflow_test<philox4x32_w30>();

    std::cout << "void counter_overflow_test() [Engine = philox4x64_w5]";
    err += counter_overflow_test<philox4x64_w5>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64_w15]";
    err += counter_overflow_test<philox4x64_w15>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64_w18]";
    err += counter_overflow_test<philox4x64_w18>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64_w25]";
    err += counter_overflow_test<philox4x64_w25>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64_w49]";
    err += counter_overflow_test<philox4x64_w49>();

    std::cout << "void discard_overflow_test() [Engine = philox4x32_w5]";
    err += discard_overflow_test<philox4x32_w5>();
    std::cout << "void discard_overflow_test() [Engine = philox4x32_w15]";
    err += discard_overflow_test<philox4x32_w15>();
    std::cout << "void discard_overflow_test() [Engine = philox4x32_w18]";
    err += discard_overflow_test<philox4x32_w18>();
    std::cout << "void discard_overflow_test() [Engine = philox4x32_w30]";
    err += discard_overflow_test<philox4x32_w30>();

    std::cout << "void discard_overflow_test() [Engine = philox4x64]";
    err += discard_overflow_test<ex::philox4x64>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64_w5]";
    err += discard_overflow_test<philox4x64_w5>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64_w15]";
    err += discard_overflow_test<philox4x64_w15>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64_w18]";
    err += discard_overflow_test<philox4x64_w18>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64_w25]";
    err += discard_overflow_test<philox4x64_w25>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64_w49]";
    err += discard_overflow_test<philox4x64_w49>();

    return TestUtils::done();
}

/*
 * All functions below have the following return values:
 *           0 in case of success
 *           1 in case of failure
 */

template <typename Engine>
int
seed_test()
{
    for (int i = 1; i < 5; i++)
    { // make sure that the state is reset properly for all idx positions
        Engine engine;
        typename Engine::result_type res;
        for (int j = 0; j < i - 1; j++)
        {
            engine();
        }
        res = engine();
        engine.seed();
        for (int j = 0; j < i - 1; j++)
        {
            engine();
        }
        if (res != engine())
        {
            std::cout << " failed while generating " << i << " elements" << std::endl;
            return 1;
        }
    }

    std::cout << " passed" << std::endl;
    return 0;
}

template <typename Engine>
int
discard_test()
{
    int ret = 0;

    constexpr std::size_t n = 10; // arbitrary length we want to check
    typename Engine::result_type reference[n];
    Engine engine;
    for (int i = 0; i < n; i++)
    {
        reference[i] = engine();
    }
    for (int i = 0; i < n; i++)
    {
        engine.seed();
        engine.discard(i);
        for (std::size_t j = i; j < n; j++)
        {
            if (reference[j] != engine())
            {
                std::cout << " failed with error in element " << j << " discard " << i << std::endl;
                ret++;
                break;
            }
        }
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < i; j++)
        {
            engine.seed();
            for (std::size_t k = 0; k < i - j; k++)
            {
                engine();
            }
            engine.discard(j);
            if (reference[i] != engine())
            {
                std::cout << " failed on step " << i << " " << j << std::endl;
                ret++;
                break;
            }
        }
    }

    if (!ret)
    {
        std::cout << " passed" << std::endl;
    }

    return ret;
}

template <typename Engine>
int
set_counter_conformance_test()
{
    Engine engine;
    std::array<typename Engine::result_type, Engine::word_count> counter;
    for (int i = 0; i < Engine::word_count - 1; i++)
    {
        counter[i] = 0;
    }

    counter[Engine::word_count - 1] = 2499; // to get 10'000 element
    engine.set_counter(counter);

    for (int i = 0; i < Engine::word_count - 1; i++)
    {
        engine();
    }

    typename Engine::result_type reference;
    if (std::is_same_v<Engine, ex::philox4x32>)
    {
        reference = 1955073260;
    }
    else
    {
        reference = 3409172418970261260;
    }
    if (engine() == reference)
    {
        std::cout << " passed" << std::endl;
        return 0;
    }
    else
    {
        std::cout << " failed" << std::endl;
        return 1;
    }
}

template <typename Engine>
int
skip_test()
{
    using T = typename Engine::result_type;
    for (T i = 1; i <= Engine::word_count + 1; i++)
    {
        Engine engine1;
        std::array<T, Engine::word_count> counter = {0, 0, 0, i / Engine::word_count};
        engine1.set_counter(counter);
        for (T j = 0; j < i % Engine::word_count; j++)
        {
            engine1();
        }

        Engine engine2;
        engine2.discard(i);

        if (engine1() != engine2())
        {
            std::cout << " failed for " << i << " skip" << std::endl;
            return 1;
        }
    }

    std::cout << " passed" << std::endl;
    return 0;
}

template <typename Engine>
int
counter_overflow_test()
{
    using T = typename Engine::result_type;
    Engine engine1;
    std::array<T, Engine::word_count> counter;
    for (int i = 0; i < Engine::word_count; i++)
    {
        counter[i] = std::numeric_limits<T>::max();
    }

    engine1.set_counter(counter);
    for (int i = 0; i < Engine::word_count; i++)
    {
        engine1();
    } // all counters overflowed == start from 0 0 0 0

    Engine engine2;

    if (engine1() == engine2())
    {
        std::cout << " passed" << std::endl;
        return 0;
    }
    else
    {
        std::cout << " failed" << std::endl;
        return 1;
    }
}

template <typename Engine>
int
discard_overflow_test()
{
    using T = typename Engine::result_type;
    using scalar_type = typename Engine::scalar_type;

    Engine engine1;
    std::array<T, Engine::word_count> counter;

    for (int i = 0; i < Engine::word_count; i++)
    {
        counter[i] = 0;
    }

    // overflow of the 0th counter element
    counter[2] = 1;

    engine1.set_counter(counter);

    Engine engine2;

    for (int i = 0; i < Engine::word_count; i++)
    {
        engine2();
    }
    for (int i = 0; i < Engine::word_count; i++)
    {
        engine2.discard(std::numeric_limits<unsigned long long>::max() &
                        (~scalar_type(0) >> (std::numeric_limits<scalar_type>::digits - Engine::word_size)));
    }

    if (engine1() == engine2())
    {
        std::cout << " passed" << std::endl;
        return 0;
    }
    else
    {
        std::cout << " failed" << std::endl;
        return 1;
    }
}
