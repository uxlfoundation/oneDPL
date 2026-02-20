# Philox RNG Productization Strategy
### [CPP reference documentation](https://en.cppreference.com/w/cpp/numeric/random/philox_engine)


## Current state
- Stated as an experimental feature.
- Added in the onAPI spec.
- Testing coverage:
    | Test | Engines Tested |
    |-----------|----------------|
    | `philox_test.pass.cpp` - KAT test checking 10000th value | `philox2x32`, `philox2x64`, `philox2x32_vec`, `philox2x64_vec`, `philox4x32`, `philox4x64`, `philox4x32_vec`, `philox4x64_vec` |
    | `engine_device_test.pass.cpp` - Device execution when objects were constructed by copy | `philox4x32`, `philox4x32_vec`, `philox4x64`, `philox4x64_vec` |
    | `engines_methods.pass.cpp` - Test of various class methods (`<<`, `>>`, `discard`, `seed`, etc.) | `philox4x32`, `philox4x32_vec`, `philox4x64`, `philox4x64_vec` |
    | `philox_uniform_real_distr_dp_test.pass.cpp` and `philox_uniform_real_distr_sp_test.pass.cpp` - Statistical test for single and double-precision distributions over philox_engine | `philox2x32`, `philox2x64`, `philox2x32_w{5,15,18,30}`, `philox2x64_w{5,15,18,25,49}`, `philox4x32_w{5,15,18,30}`, `philox4x64_w{5,15,18,25,49}` |
- Philox is planned to be part of C++26, compilers don't support it officially yet, but it's available when using Clang's trunk ([sample](https://godbolt.org/z/Pr4d7qeq4)).


## Proposed steps
- Check the correspondence with the latest proposal and the existing corrections ([P2075R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2075r6.pdf), [issue4134](https://cplusplus.github.io/LWG/issue4134), [issue4153](https://cplusplus.github.io/LWG/issue4153)).
- Correct the I/O operator, some initial work already done in [dev/etyulene/philox_iostreams_corrections](https://github.com/uxlfoundation/oneDPL/tree/dev/etyulene/philox_iostreams_corrections).
- Generate KAT vectors for some `w`, `!=32` and `!=64` using Clang trunk and extend the testing (10000th elements).
- Remove experimental namespace.
- [???] Review the existing implementations in GCC and Clang to see the principal differences in the implementation.
- [???] Add testing of the engine to oneCI.
- [Optional] Consider performance improvements.
