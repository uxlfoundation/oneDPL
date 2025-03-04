##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

option(ONEDPL_TEST_ENABLE_KT_ESIMD "Enable ESIMD-based kernel template tests")
option(ONEDPL_TEST_ENABLE_KT_SYCL "Enable SYCL-based kernel template tests")

function(_generate_test _target_name _test_path)
    add_executable(${_target_name} EXCLUDE_FROM_ALL ${_test_path})

    target_link_libraries(${_target_name} PRIVATE oneDPL)
    set_target_properties(${_target_name} PROPERTIES CXX_EXTENSIONS NO)

    add_test(NAME ${_target_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_target_name})
    set_tests_properties(${_target_name} PROPERTIES SKIP_RETURN_CODE 77)
    if (DEFINED DEVICE_SELECTION_LINE)
        set_tests_properties(${_target_name} PROPERTIES ENVIRONMENT ${DEVICE_SELECTION_LINE})
    endif()

    add_custom_target(run-${_target_name}
        COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_target_name}$$ --output-on-failure --no-label-summary
        USES_TERMINAL
        DEPENDS ${_target_name}
        COMMENT "Build and run test ${_target_name}")
endfunction(_generate_test)

function(_generate_test_randomly _target_name _test_path _probability_permille)
    string(RANDOM LENGTH 3 ALPHABET "0123456789" _random)
    if ((${_random} LESS ${_probability_permille}) AND (NOT TARGET ${_target_name}))
        _generate_test(${_target_name} ${_test_path})
    endif()
endfunction()

# Arguments with special treatment:
#   _val_type. Treated as absent for values: "" (empty string).
#              If absent, key sort test is generated, otherwise key-value sort test
function(_generate_esimd_sort_test _base_file _data_per_work_item _work_group_size _key_type _val_type _probability_permille)
    if ((NOT TARGET build-esimd-tests) AND (NOT TARGET run-esimd-tests))
        add_custom_target(build-esimd-tests COMMENT "Build all ESIMD-based kernel template tests")
        add_custom_target(run-esimd-tests
            COMMAND "${CMAKE_CTEST_COMMAND}" -R "^run-esimd-tests$" --output-on-failure --no-label-summary
            DEPENDS build-esimd-tests
            COMMENT "Build and run all ESIMD tests")
    endif()

    string(REPLACE "_t" "" _key_type_short ${_key_type})
    if (NOT "${_val_type}" STREQUAL "")
        string(REPLACE "_t" "" _val_type_short ${_val_type})
        set(_target_name "${_base_file}_dpwi${_data_per_work_item}_wgs${_work_group_size}_${_key_type_short}_${_val_type_short}")
        set(_test_path "${_base_file}.cpp")
    else()
        set(_target_name "${_base_file}_dpwi${_data_per_work_item}_wgs${_work_group_size}_${_key_type_short}")
        set(_test_path "${_base_file}.cpp")
    endif()

    _generate_test_randomly(${_target_name} ${_test_path} ${_probability_permille})
    if(TARGET ${_target_name})
        add_dependencies(build-esimd-tests ${_target_name})
        add_dependencies(run-esimd-tests ${_target_name})

        target_compile_definitions(${_target_name} PRIVATE TEST_DATA_PER_WORK_ITEM=${_data_per_work_item})
        target_compile_definitions(${_target_name} PRIVATE TEST_WORK_GROUP_SIZE=${_work_group_size})
        target_compile_definitions(${_target_name} PRIVATE TEST_KEY_TYPE=${_key_type})
        if (NOT "${_val_type}" STREQUAL "")
            target_compile_definitions(${_target_name} PRIVATE TEST_VALUE_TYPE=${_val_type})
        endif()
    endif()
endfunction()

# Tests for ESIMD-based Kernel Templates are randomly generated.
# Purpose:
# - Cover as much as possible cases with limited resources by running different sets during each session.
# - Get more fine-grained results and thus better debuggability.
# - Allow parallel compilation and thus speed up testing process.
# - Provide convenient compilation and execution times of an individual test during development.
#
# Strategy: generate a test randomly basing on a subset of its parameters.
# All possible parameters:
# 1. Kernel Template configuration:
#     - data_per_workitem
#     - workgroup_size
# 2. Algorithm-specific parameters:
#     - sorting order
#     - radix_bits
# 3. Input sequence type (e.g. sycl::buffer, USM and etc.).
# 4. Input data type: key type or a combination of key-value types.
# 5. Number of elements.
# Parameters #1 and #4 were selected for generation to allow compilation and execution of a test within ~5 minutes
function(_generate_esimd_sort_tests _key_value_pairs)
    set(_base_file_all "esimd_radix_sort" "esimd_radix_sort_out_of_place")
    set(_base_file_by_key_all "esimd_radix_sort_by_key" "esimd_radix_sort_by_key_out_of_place")
    set(_data_per_work_item_all "32" "64" "96" "128" "160" "192" "224" "256" "288" "320" "352" "384" "416" "448" "480" "512")
    set(_work_group_size_all "64")
    set(_type_all "char" "uint16_t" "int" "uint64_t" "float" "double")

    foreach (_data_per_work_item ${_data_per_work_item_all})
        foreach (_work_group_size ${_work_group_size_all})
            foreach (_key_type ${_type_all})
                if(_key_value_pairs)
                    foreach (_base_file ${_base_file_by_key_all})
                        foreach (_val_type ${_type_all})
                            # ~1-2 tests: (2/1000) * (16 * 6 * 6 * 2)
                            _generate_esimd_sort_test(${_base_file} ${_data_per_work_item} ${_work_group_size} ${_key_type} ${_val_type} 2)
                        endforeach()
                    endforeach()
                else()
                    foreach (_base_file ${_base_file_all})
                        # ~1-2 test: (10/1000) * (16 * 6 * 2)
                        _generate_esimd_sort_test(${_base_file} ${_data_per_work_item} ${_work_group_size} ${_key_type} "" 10)
                    endforeach()
                endif()
            endforeach()
        endforeach()
    endforeach()
endfunction()

if (ONEDPL_TEST_ENABLE_KT_ESIMD)
    _generate_esimd_sort_tests(FALSE) # esimd_radix_sort, random
    _generate_esimd_sort_tests(TRUE) # esimd_radix_sort_by_key, random

    # Pin some cases to track them
    _generate_esimd_sort_test("esimd_radix_sort_by_key_out_of_place" "96" "64" "uint32_t" "uint32_t" 1000)
    _generate_esimd_sort_test("esimd_radix_sort" "384" "64" "int32_t" "" 1000)
endif()

function (_generate_gpu_scan_test _data_per_work_item _work_group_size _type _probability_permille)
    if ((NOT TARGET build-scan-kt-tests) AND (NOT TARGET run-scan-kt-tests))
        add_custom_target(build-scan-kt-tests COMMENT "Build all scan kernel template tests")
        add_custom_target(run-scan-kt-tests
            COMMAND "${CMAKE_CTEST_COMMAND}" -R "^run-scan-kt-tests$" --output-on-failure --no-label-summary
            DEPENDS build-scan-kt-tests
            COMMENT "Build and run all scan kernel template tests")
    endif()

    string(REPLACE "_t" "" _type_short ${_type})
    set(_target_name "single_pass_scan_dpwi${_data_per_work_item}_wgs${_work_group_size}_${_type_short}")
    set(_test_path "single_pass_scan.cpp")

    _generate_test_randomly(${_target_name} ${_test_path} ${_probability_permille})
    if(TARGET ${_target_name})
        add_dependencies(build-scan-kt-tests ${_target_name})
        add_dependencies(run-scan-kt-tests ${_target_name})

        target_compile_definitions(${_target_name} PRIVATE TEST_DATA_PER_WORK_ITEM=${_data_per_work_item})
        target_compile_definitions(${_target_name} PRIVATE TEST_WORK_GROUP_SIZE=${_work_group_size})
        target_compile_definitions(${_target_name} PRIVATE TEST_TYPE=${_type})
    endif()
endfunction()

function(_generate_gpu_scan_tests)
    set(_data_per_work_item_all "1" "2" "4" "8" "16" "32")
    set(_work_group_size_all "64" "128" "256" "512" "1024")
    set(_type_all "uint32_t" "int32_t" "float" "int64_t" "uint64_t" "double")

    foreach (_data_per_work_item ${_data_per_work_item_all})
        foreach (_work_group_size ${_work_group_size_all})
            foreach (_type ${_type_all})
                # ~1-2 test: (10/1000) * (6 * 5 * 6)
                _generate_gpu_scan_test(${_data_per_work_item} ${_work_group_size} ${_type} 10)
            endforeach()
        endforeach()
    endforeach()
endfunction()

if (ONEDPL_TEST_ENABLE_KT_SYCL)
    _generate_gpu_scan_tests()

    # Pin some cases to track them
    _generate_gpu_scan_test("8" "512" "uint32_t" 1000)
    _generate_gpu_scan_test("4" "256" "float" 1000)
    _generate_gpu_scan_test("16" "64" "int64_t" 1000)
endif()
