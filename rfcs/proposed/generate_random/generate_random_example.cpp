// -*- C++ -*-
//===-- generate_random_example.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runnable companion to README.md. Checks no-policy engine sequence equivalence
// and basic output properties. Exits 0 only if all checks pass.
//
// Build:
//   icpx -fsycl -std=c++20 -O2 -I <onedpl>/include generate_random_example.cpp -o gr
// Add the AVX-512 Philox fast path:  -mavx512f
// Add host thread parallelism:       -ltbb
// Policy-generated sequences may vary between configurations.
//
// -fsycl is optional: <oneapi/dpl/random> also builds with a plain C++20 compiler
// (g++ / clang++), see open question 2 in README.md.
//
#include <oneapi/dpl/execution>
#include <oneapi/dpl/random>

#include <iostream>
#include <vector>

int
main()
{
    namespace ex = oneapi::dpl::execution;
    using engine_t = oneapi::dpl::philox4x32; // P2075
    using word_t = engine_t::result_type;      // uint_fast32_t, i.e. 64-bit on LP64
    constexpr std::size_t n = 1'000'000;
    bool ok = true;
    auto report = [&ok](const char* what, bool good) {
        std::cout << (good ? "[ ok ] " : "[FAIL] ") << what << "\n";
        ok = ok && good;
    };

    // Reference: what `for (auto& e : v) e = g();` produces.
    std::vector<word_t> ref(n);
    engine_t g_ref(777);
    for (auto& e : ref)
        e = g_ref();

    // 1. C++26: fill a range from an engine. Required to be bit-identical.
    std::vector<word_t> v(n);
    engine_t g1(777);
    oneapi::dpl::ranges::generate_random(v, g1);
    report("engine fill == scalar loop", v == ref);

    // 2. C++26: fill a range through a distribution.
    std::vector<float> f(n);
    engine_t g2(777);
    oneapi::dpl::uniform_real_distribution<float> d(1.0f, 2.0f);
    oneapi::dpl::ranges::generate_random(f, g2, d);
    report("uniform_real values in [1, 2)", f[0] >= 1.0f && f[0] < 2.0f && f[n - 1] >= 1.0f && f[n - 1] < 2.0f);

    // 3. oneDPL extension: policy output need not match the scalar sequence.
    // The final state of g3 is valid but unspecified.
    std::vector<word_t> p(n);
    engine_t g3(777);
    auto last = oneapi::dpl::ranges::generate_random(ex::par_unseq, p, g3);
    bool policy_ok = last == p.end();
    for (auto value : p)
        policy_ok = policy_ok && value >= engine_t::min() && value <= engine_t::max();
    report("par_unseq output range and engine bounds", policy_ok);

    std::cout << (ok ? "PASSED\n" : "FAILED\n");
    return ok ? 0 : 1;
}
