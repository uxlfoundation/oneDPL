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
add_subdirectory(kt)

# rng_tests
set (ranlux_24_48_test.pass_timeout_debug "900") # 15min
set (ranlux_24_48_test.pass_timeout_release "720") # 12min

# If TBB headers are available, libstdc++ enables TBB backend; some TBB symbols become undefined unless it is linked.
# The undefined symbol error arises only with GCC compiler.
# There is no known way to limit the workaround to libstdc++ only.
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    if (ONEDPL_BACKEND MATCHES "^(serial|omp|dpcpp_only)$")
        find_package(TBB 2021 QUIET COMPONENTS tbb OPTIONAL_COMPONENTS tbbmalloc)
        if (TBB_FOUND)
            message(STATUS "Tests are linked against TBB to avoid undefined symbol errors due to TBB usage in libstdc++")
            set(_gcc_link_tbb TRUE)
        endif()
    endif()
endif()

# disable fast math for ALL TESTS
# icpx and dpcpp accept -fno-fast-math, but icx-cl and dpcpp-cl only accept /clang:-fno-fast-math
foreach(_fno_fast_math_flag -fno-fast-math /clang:-fno-fast-math)
    string(MAKE_C_IDENTIFIER ${_fno_fast_math_flag} FLAG_DISPLAY_NAME)
    check_cxx_compiler_flag(${_fno_fast_math_flag} ${FLAG_DISPLAY_NAME}_option)
    if (${FLAG_DISPLAY_NAME}_option)
        target_compile_options(oneDPL INTERFACE ${_fno_fast_math_flag})
        set(_fno_fast_math_option ${_fno_fast_math_flag})
        break()
    endif()
endforeach()
if (_fno_fast_math_option)
    add_compile_options(${_fno_fast_math_option})
else()
    message(STATUS "oneDPL: -fno-fast-math is not supported by current compiler")
endif()

# usage of Kernel names in tests
set(ONEDPL_TEST_EXPLICIT_KERNEL_NAMES AUTO CACHE STRING "Use explicit Kernel names in tests")
set_property(CACHE ONEDPL_TEST_EXPLICIT_KERNEL_NAMES PROPERTY STRINGS AUTO ALWAYS)
message(STATUS "Usage of Kernel names in tests: ONEDPL_TEST_EXPLICIT_KERNEL_NAMES=${ONEDPL_TEST_EXPLICIT_KERNEL_NAMES}")
if (${ONEDPL_TEST_EXPLICIT_KERNEL_NAMES} STREQUAL AUTO)
    message(STATUS "    Usage of Kernel names in tests depend on unnamed lambda option")
    # State of TEST_EXPLICIT_KERNEL_NAMES will be definel later in C++ code
elseif(${ONEDPL_TEST_EXPLICIT_KERNEL_NAMES} STREQUAL ALWAYS)
    add_definitions(-DTEST_EXPLICIT_KERNEL_NAMES=1)
    message(STATUS "    Use Kernel names in tests :")
    message(STATUS "        #define TEST_EXPLICIT_KERNEL_NAMES 1")
else()
    message(FATAL_ERROR "Unsupported explicit Kernel names usage in tests: ${ONEDPL_TEST_EXPLICIT_KERNEL_NAMES}.\n"
            "Select one of the following values: AUTO, ALWAYS")
endif()

add_custom_target(build-onedpl-tests
    COMMENT "Build all oneDPL tests")

add_custom_target(run-onedpl-tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    USES_TERMINAL
    DEPENDS build-onedpl-tests
    COMMENT "Build and run all oneDPL tests")

macro(onedpl_construct_exec test_source_file _test_name switch_off_checked_iterators custom_define extra_test_label)
    # Disable checked iterators on Windows for debug mode
    # For details please see
    #    https://learn.microsoft.com/en-us/cpp/standard-library/iterator-debug-level?view=msvc-170
    #    https://learn.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library?view=msvc-170
    #    https://stackoverflow.com/questions/51494506/replacing-md-with-mt-in-cmake-is-not-possible-in-release-mode
    if (WIN32 AND ${_build_type_in_lower} STREQUAL "debug" AND ${switch_off_checked_iterators})
        set(_use_release_in_debug 1)
    else()
        set(_use_release_in_debug 0)
    endif()

    if (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off AND _use_release_in_debug)
        # link with release runtime is not supported for debug PSTL offload while one is requested, skip
    else()
        add_executable(${_test_name} EXCLUDE_FROM_ALL "${test_source_file}")
        target_compile_definitions(${_test_name} PRIVATE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)

        if (NOT ${custom_define} STREQUAL "")
            target_compile_definitions(${_test_name} PRIVATE ${custom_define})
        endif()
        if (MSVC)
            target_compile_options(${_test_name} PRIVATE /bigobj)
        endif()

        if (_use_release_in_debug)
            target_compile_definitions(${_test_name} PRIVATE _ITERATOR_DEBUG_LEVEL=0)
            target_compile_options(${_test_name} PRIVATE "/MD$<$<CONFIG:Debug>:>")
        endif()

        # oneDPL test harness may initialize a C++ iterator using iterator with different type
        # that may break code when using Intel(R) C++ Compiler Classic with -O3 flag on Linux
        if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL Intel)
            target_compile_options(${_test_name} PRIVATE $<$<CONFIG:Release>:-fno-strict-aliasing>)
        endif()

        target_include_directories(${_test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        target_link_libraries(${_test_name} PRIVATE oneDPL)
        if (_gcc_link_tbb)
            target_link_libraries(${_test_name} PRIVATE TBB::tbb)
        endif()
        set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)

        add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})

        if (DEFINED ${_test_name}_timeout_${_build_type_in_lower})
            message(STATUS
            "Timeout for ${_test_name} is set to ${${_test_name}_timeout_${_build_type_in_lower}}, "
            "it overrides user-defined timeout for CTest")
            set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${${_test_name}_timeout_${_build_type_in_lower}})
        elseif (DEFINED ${_test_name}_timeout_release)
            # Apply default (release) timeout
            message(STATUS
            "Timeout for ${_test_name} is set to ${${_test_name}_timeout_release}, "
            "it overrides user-defined timeout for CTest")
            set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${${_test_name}_timeout_release})
        endif()

        set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)

        if (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
            if (DEFINED DEVICE_SELECTION_LINE)
                message(FATAL_ERROR "PSTL offload tests ${_test_name} don't need ${DEVICE_SELECTION_LINE} variable, so one should be not set.")
            endif()
            # This test is only one requiring device selection environment, set it manually.
            if ("${_test_name}" STREQUAL selected_different_device.pass AND _ONEDPL_PSTL_OFFLOAD MATCHES "(cpu|gpu)")
                set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT "${DEVICE_SELECTION_VARIABLE}=${ONEDPL_DEVICE_BACKEND}:${_ONEDPL_PSTL_OFFLOAD}")
            endif()
        elseif (DEFINED DEVICE_SELECTION_LINE)
            file(GLOB_RECURSE DS_TESTS "parallel_api/dynamic_selection/*.cpp")
            if (NOT "${DS_TESTS}" MATCHES "${_test_name}") # do not set SYCL_DEVICE_FILTER for dynamic selection tests
                set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT ${DEVICE_SELECTION_LINE})
            endif()
        endif()

        add_custom_target(run-${_test_name}
            COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_test_name}$$ --output-on-failure --no-label-summary
            USES_TERMINAL
            DEPENDS ${_test_name}
            COMMENT "Build and run test ${_test_name}")

        # Add labels and group targets
        set(_test_labels "")

        file(RELATIVE_PATH _test_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${test_source_file})
        get_filename_component(_test_rel_path ${_test_rel_path} DIRECTORY)
        if (_test_rel_path)
            string(REPLACE "/" ";" _test_labels ${_test_rel_path})
        endif()

        list(APPEND _test_labels ${extra_test_label})

        if (_test_labels)
            set_tests_properties(${_test_name} PROPERTIES LABELS "${_test_labels}")
        endif()

        foreach (_label ${_test_labels})
            if (NOT TARGET build-onedpl-${_label}-tests)
                add_custom_target(build-onedpl-${_label}-tests COMMENT "Build tests with label ${_label}")

                add_custom_target(run-onedpl-${_label}-tests
                    COMMAND "${CMAKE_CTEST_COMMAND}" -L ^${_label}$$ --output-on-failure --no-label-summary
                    USES_TERMINAL
                    DEPENDS build-onedpl-${_label}-tests
                    COMMENT "Build and run tests with label ${_label}")
            endif()
            add_dependencies(build-onedpl-${_label}-tests ${_test_name})
        endforeach()
        add_dependencies(build-onedpl-tests ${_test_name})

    endif()
endmacro()

macro(onedpl_add_test test_source_file switch_off_checked_iterators)
    set(_build_type_in_lower "unknown")
    if (CMAKE_BUILD_TYPE)
        string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type_in_lower)
    endif()

    get_filename_component(_test_name ${test_source_file} NAME)
    string(REPLACE "\.cpp" "" _test_name ${_test_name})

    set(coal_tests "reduce.pass" "transform_reduce.pass" "count.pass" "sycl_iterator_reduce.pass" "minmax_element.pass")
    set(workaround_for_igpu_64bit_reduction_tests "reduce_by_segment.pass")
    # mark those tests with pstloffload_smoke_tests label
    set (pstloffload_smoke_tests "adjacent_find.pass" "copy_move.pass" "merge.pass" "partial_sort.pass" "remove_copy.pass"
        "transform_reduce.pass" "transform_reduce.pass.coal" "transform_scan.pass" "algorithm.pass"
        "execution.pass" "functional.pass" "algorithms_redirection.pass" "usm_memory_replacement.pass")

    set(extra_test_label "")
    if (_test_name IN_LIST pstloffload_smoke_tests)
        set(extra_test_label "pstloffload_smoke_tests")
    endif()

    if (_test_name IN_LIST coal_tests)
        onedpl_construct_exec(${test_source_file} ${_test_name} ${switch_off_checked_iterators} "-D_ONEDPL_DETECT_SPIRV_COMPILATION=1" "${extra_test_label}")
        onedpl_construct_exec(${test_source_file} ${_test_name}.coal ${switch_off_checked_iterators} "-D_ONEDPL_DETECT_SPIRV_COMPILATION=0" "${extra_test_label}")
    elseif (_test_name IN_LIST workaround_for_igpu_64bit_reduction_tests)
        onedpl_construct_exec(${test_source_file} ${_test_name} ${switch_off_checked_iterators} "" "${extra_test_label}")
        string(REPLACE "\.pass" "_workaround_64bit_reduction\.pass" _test_name ${_test_name})
        onedpl_construct_exec(${test_source_file} ${_test_name} ${switch_off_checked_iterators} "-D_ONEDPL_TEST_FORCE_WORKAROUND_FOR_IGPU_64BIT_REDUCTION=1" "${extra_test_label}")
    elseif(_test_name STREQUAL "free_after_unload.pass")
        onedpl_construct_exec(${test_source_file} ${_test_name} ${switch_off_checked_iterators} "" "${extra_test_label}")
        onedpl_construct_exec(${test_source_file} ${_test_name}.after_pstl_offload ${switch_off_checked_iterators} "" "${extra_test_label}")
    else()
        onedpl_construct_exec(${test_source_file} ${_test_name} ${switch_off_checked_iterators} "" "${extra_test_label}")
    endif()
endmacro()

set(_regexp_dpcpp_backend_required "(xpu_api/ranges|xpu_api/algorithms|test/xpu_api/cmath/nearbyint|test/xpu_api/containers|xpu_api/functional|test/xpu_api/iterators|test/xpu_api/language.support/support.initlist|test/xpu_api/language.support/support.types|xpu_api/numerics|test/xpu_api/random|test/xpu_api/ratio|test/xpu_api/tuple|test/xpu_api/utilities|parallel_api/dynamic_selection/sycl)")
set(_regexp_switch_off_checked_it  "(test/general/header_order_ranges|test/parallel_api/algorithm/alg.sorting/alg.min.max|test/parallel_api/ranges|test/parallel_api/dynamic_selection|test/xpu_api/iterators/iterator.primitives|test/xpu_api/random/device_tests|test/xpu_api/random/interface_tests|test/xpu_api/random/statistics_tests|test/parallel_api/numeric/numeric.ops/transform_scan)")

set(_regexp_pstl_offload_only "(test/pstl_offload)")
set(_regexp_pstl_offload_parallel_api "(test/xpu_api|test/parallel_api/algorithm|test/parallel_api/memory|test/parallel_api/numeric)")
set(_regexp_pstl_offload_exclude "(test/parallel_api/algorithm/alg.sorting/alg.binary.search|test/parallel_api/numeric/numeric.ops/.*_by_segment)")

# Some of the headers tests C++20/23 files that can or cannot be available in C++17 mode
# So we need to check file presence and skip the test if no header available
# The test should be placed in the special folder and the name should be the same as the header that is required to be present
set(_regexp_pstl_offload_header_presence_required "(test/pstl_offload/headers/headers.standard/requires_header_presence)")

file(GLOB_RECURSE UNIT_TESTS "*.pass.cpp")
foreach (_file IN LISTS UNIT_TESTS)
    if (_file MATCHES "${_regexp_pstl_offload_only}")
        if (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
            if (_file MATCHES "${_regexp_pstl_offload_header_presence_required}")
                get_filename_component(_filename "${_file}" NAME_WE)
                CHECK_INCLUDE_FILE_CXX(${_filename} _has_header_file)
                if (_has_header_file)
                    onedpl_add_test(${_file} false)
                endif()
                unset(_has_header_file CACHE)
            else()
                onedpl_add_test(${_file} false)
            endif()
        else()
            continue()
        endif()
    elseif (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
        if (_file MATCHES "${_regexp_pstl_offload_parallel_api}" AND NOT _file MATCHES "${_regexp_pstl_offload_exclude}")
            onedpl_add_test(${_file} true)
        endif()
    else()
        if (_file MATCHES "${_regexp_dpcpp_backend_required}")
            if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
                # Switch off checked iterators
                onedpl_add_test(${_file} true)
            else()
                message(STATUS "Skip test ${_file} as it requires DPC++ backend")
            endif()
        else()
            if (_file MATCHES "${_regexp_switch_off_checked_it}")
                # Switch off checked iterators
                onedpl_add_test(${_file} true)
            else()
                # Do not switch off checked iterators
                onedpl_add_test(${_file} false)
            endif()
        endif()
    endif()

endforeach()

# add additional TUs if required
if (TARGET multiple_translation_units.pass)
    target_sources(
        multiple_translation_units.pass PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/general/multiple_translation_units/translation_unit1.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/general/multiple_translation_units/translation_unit2.cpp")
endif()
