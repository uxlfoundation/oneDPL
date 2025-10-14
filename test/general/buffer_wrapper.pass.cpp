#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
#include <vector>

using namespace TestUtils;
//This macro is required for the tests to work correctly in CI with tbb-backend.
#if TEST_DPCPP_BACKEND_PRESENT
struct test_buffer_wrapper
{
    template <typename Iterator, typename T>
    void operator()(Iterator begin, Iterator end, T* expected_data, std::size_t size)
    {
        EXPECT_TRUE(begin == begin, "operator == returned false negative");
        EXPECT_TRUE(!(begin == begin + 1), "operator == returned false positive");

        EXPECT_TRUE(begin != begin + 1, "operator != returned false negative");
        EXPECT_TRUE(!(begin != begin), "operator != returned false positive");

        auto it = begin;

        EXPECT_TRUE(it == begin, "wrong effect of iterator's copy constructor");

        it = end;

        EXPECT_TRUE(it == end, "wrong effect of iterator's copy assignment operator");
        EXPECT_TRUE(it - size == begin, "wrong effect of iterator's operator - integer");
        EXPECT_TRUE(begin + size == end, "wrong effect of iterator's operator + integer");
        EXPECT_TRUE(end - begin == size, "wrong effect of iterator's operator - iterator");

        auto buf = begin.get_buffer();
        sycl::host_accessor buf_accessor(buf, sycl::read_only);
        auto actual_data = buf_accessor.get_pointer();

        EXPECT_EQ(expected_data, actual_data, "wrong effect of iterator's method get_buffer");
    }
};
#endif

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    std::size_t size = 1000;
    sycl::buffer<std::uint32_t> buf{size};
    test_buffer_wrapper test{};
    sycl::host_accessor buf_accessor(buf, sycl::read_only);
    auto data_ptr = buf_accessor.get_pointer();

    test(oneapi::dpl::begin(buf), oneapi::dpl::end(buf), data_ptr, size);
    test(oneapi::dpl::begin(buf, sycl::write_only), oneapi::dpl::end(buf, sycl::write_only), data_ptr, size);
    test(oneapi::dpl::begin(buf, sycl::write_only, sycl::property::no_init{}),
         oneapi::dpl::end(buf, sycl::write_only, sycl::property::no_init{}), data_ptr, size);
    test(oneapi::dpl::begin(buf, sycl::property::no_init{}),
         oneapi::dpl::end(buf, sycl::property::no_init{}), data_ptr, size);

#endif
    return done(TEST_DPCPP_BACKEND_PRESENT);
}
