//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#include <oneapi/dpl/dynamic_selection>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

// stb/*.h files can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/stb/*.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;
using namespace sycl;
namespace ex = oneapi::dpl::experimental;

// Few useful acronyms.
constexpr auto sycl_read = access::mode::read;
constexpr auto sycl_write = access::mode::write;
constexpr auto sycl_device = access::target::device;

int g_num_images = 4;

const char* g_fnames[3] = {"input/silver512.png", "input/nahelam512.bmp", "input/silverfalls1.png"};
int g_width[4] = {0, 0, 0, 0};
int g_height[4] = {0, 0, 0, 0};
int g_channels[4] = {0, 0, 0, 0};

void
fillVectors(int mix, std::vector<size_t>& num_pixels, std::vector<sycl::buffer<uint8_t>>& input_buffers,
            std::vector<sycl::buffer<uint8_t>>& output_buffers)
{

    if (mix > 3)
        g_num_images = 3;
    else
        g_num_images = 4;

    for (int i = 0; i < g_num_images; ++i)
    {
        int img_width, img_height, channels;
        int index = 0;
        switch (mix)
        {
        case 1:
            // 1 - Small images only
            index = i % 2;
            break;
        case 2:
            // 2 - Large images only
            index = 2;
            break;
        case 3:
            // 3 - 2 small : 2 large
            index = std::min(i % 4, 2);
            break;
        case 4:
            // 4 - 2 small : 1 large
            index = i % 3;
            break;
        case 5:
            // 5 - 1 small : 2 large
            index = std::min(i % 3 + 1, 2);
            break;
        }

        uint8_t* image = stbi_load(g_fnames[index], &img_width, &img_height, &channels, 0);
        if (image == NULL)
        {
            cout << "Error in loading the image " << g_fnames[index] << "\n";
            exit(1);
        }
        g_width[i] = img_width;
        g_height[i] = img_height;
        g_channels[i] = channels;
        size_t npixels = img_width * img_height;
        size_t img_size = img_width * img_height * channels;
        input_buffers.push_back(sycl::buffer{image, sycl::range(img_size)});
        num_pixels.push_back(npixels);
        uint8_t* out_data = new uint8_t[img_size];
        memset(out_data, 0, img_size * sizeof(uint8_t));
        output_buffers.push_back(sycl::buffer{out_data, sycl::range(img_size)});
    }
}

void
writeImages(std::vector<sycl::buffer<uint8_t>>& output_buffers)
{
    const char* out_names[4] = {"out0.png", "out1.png", "out2.png", "out3.png"};
    for (int i = 0; i < g_num_images; ++i)
    {
        stbi_write_png(out_names[i], g_width[i], g_height[i], g_channels[i],
                       reinterpret_cast<uint8_t*>(output_buffers[i].get_host_access().get_pointer()),
                       g_width[i] * g_channels[i]);
    }
}

// SYCL does not need any special mark-up for functions which are called from
// SYCL kernel and defined in the same compilation unit. SYCL compiler must be
// able to find the full call graph automatically.
// always_inline as calls are expensive on Gen GPU.
// Notes:
// - coeffs can be declared outside of the function, but still must be constant
// - SYCL compiler will automatically deduce the address space for the two
//   pointers; sycl::multi_ptr specialization for particular address space
//   can used for more control
__attribute__((always_inline)) static void
ApplyFilter(multi_ptr<const unsigned char, access::address_space::global_space, (sycl::access::decorated)2> src_image,
            uint8_t* dst_image, int i)
{
    i *= 3;
    float temp;
    temp = (0.393f * src_image[i]) + (0.769f * src_image[i + 1]) + (0.189f * src_image[i + 2]);
    dst_image[i] = temp > 255 ? 255 : temp;
    temp = (0.349f * src_image[i]) + (0.686f * src_image[i + 1]) + (0.168f * src_image[i + 2]);
    dst_image[i + 1] = temp > 255 ? 255 : temp;
    temp = (0.272f * src_image[i]) + (0.534f * src_image[i + 1]) + (0.131f * src_image[i + 2]);
    dst_image[i + 2] = temp > 255 ? 255 : temp;
}

template <typename T>
int
invokeDS(int num_offloads, std::vector<sycl::queue>& resources, std::vector<size_t>& npixels,
         std::vector<sycl::buffer<uint8_t>>& input_buffers, std::vector<sycl::buffer<uint8_t>>& output_buffers)
{

    T p(resources);

    auto t_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_offloads; ++i)
    {
        try
        {
            sycl::buffer<uint8_t>& in = input_buffers[i % g_num_images];
            sycl::buffer<uint8_t>& out = output_buffers[i % g_num_images];
            size_t num_pixels = npixels[i % g_num_images];

            auto f1 = [&](sycl::queue q, size_t n) {
                // See what device was actually selected for this queue.
                cout << "Lambda running on " << q.get_device().get_info<info::device::name>() << "\n";

                // Submit a command group for execution. Returns immediately, not waiting
                // for command group completion.
                return q.submit([&](auto& h) {
                    // This lambda defines a "command group" - a set of commands for the
                    // device sharing some state and executed in-order - i.e. creation of
                    // accessors may lead to on-device memory allocation, only after that
                    // the kernel will be enqueued.
                    // A command group can contain at most one parallel_for, single_task or
                    // parallel_for_workgroup construct.
                    accessor image_acc(in, h, read_only);
                    accessor image_exp_acc(out, h, write_only);

                    // This is the simplest form sycl::handler::parallel_for -
                    // - it specifies "flat" 1D ND range(num_pixels), runtime will select
                    //   local size
                    // - kernel lambda accepts single sycl::id argument, which has very
                    //   limited API; see the spec for more complex forms
                    // the lambda parameter of the parallel_for is the kernel, which
                    // actually executes on device
                    h.parallel_for(range<1>(num_pixels), [=](auto i) {
                        ApplyFilter(image_acc.get_pointer(), image_exp_acc.get_pointer(), i);
                    });
                });
            };
            ex::submit_and_wait(p, f1, num_pixels);
        }
        catch (const sycl::exception& e)
        {
            // This catches only synchronous exceptions that happened in current thread
            // during execution. The asynchronous exceptions caused by execution of the
            // command group are caught by the asynchronous exception handler
            // registered. Synchronous exceptions are usually those which are thrown
            // from the SYCL runtime code, such as on invalid constructor arguments. An
            // example of asynchronous exceptions is error occurred during execution of
            // a kernel. Make sure sycl::exception is caught, not std::exception.
            cout << "SYCL exception caught: " << e.what() << "\n";
            return 1;
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();

    cout << "Total time == " << total_time << " us\n";
    return 0;
}

void
printUsage(char* exe_name)
{
    std::cout << "Application requires arguments. Usage: " << exe_name << " <num_images> <mix> <policy>" << std::endl
              << "Mix:" << std::endl
              << "1 - Small images only" << std::endl
              << "2 - Large images only" << std::endl
              << "3 - 2 small : 2 large" << std::endl
              << "4 - 2 small : 1 large" << std::endl
              << "5 - 1 small : 2 large" << std::endl
              << std::endl
              << "Policy:" << std::endl
              << "1 - Fixed Resource Policy (CPU)" << std::endl
              << "2 - Fixed Resource Policy (GPU)" << std::endl
              << "3 - Round Robin Policy" << std::endl
              << "4 - Dynamic Load Policy" << std::endl
              << "5 - Auto Tune Policy" << std::endl;
}

void
displayConfig(int mix, int policy, int num_offloads)
{
    std::cout << "Processing " << num_offloads << " images\n";
    switch (mix)
    {
    case 1:
        // 1 - Small images only
        std::cout << "Only small images\n";
        break;
    case 2:
        // 2 - Large images only
        std::cout << "Only large images\n";
        break;
    case 3:
        // 3 - 2 small : 2 large
        std::cout << "50/50 small images and large images\n";
        break;
    case 4:
        // 4 - 2 small : 1 large
        std::cout << "2 small images for each large image\n";
        break;
    case 5:
        // 5 - 1 small : 2 large
        std::cout << "2 large images for each small image\n";
        break;
    }
    switch (policy)
    {
    case 1:
        std::cout << "Using fixed_resource_policy to always select the CPU\n";
        break;
    case 2:
        std::cout << "Using fixed_resource_policy to always select the GPU\n";
        break;
    case 3:
        std::cout << "Using round_robin_policy to alternate between CPU and GPU\n";
        break;
    case 4:
        std::cout << "Using dynamic_load_policy to select least loaded device\n";
        break;
    case 5:
        std::cout << "Using auto_tune_policy to select best device for each image size\n";
        break;
    }
    std::cout << "\n";
}

int
main(int argc, char* argv[])
{
    int num_offloads{100}, mix{2}, policy{2};
    std::vector<sycl::queue> resources;

    std::vector<sycl::buffer<uint8_t>> input_buffers;
    std::vector<size_t> num_pixels;
    std::vector<sycl::buffer<uint8_t>> output_buffers;

    if (argc < 4)
    {
        printUsage(argv[0]);
        return -1;
    }
    else
    {
        int n = std::atoi(argv[1]);
        if (n <= 0)
        {
            std::cout << "num offloads must be a positive integer." << std::endl;
            return -1;
        }
        else
        {
            num_offloads = n;
        }
        int m = std::atoi(argv[2]);
        if (m <= 0 || m > 5)
        {
            std::cout << "Improper mix choice.\n";
            printUsage(argv[0]);
            return -1;
        }
        else
        {
            mix = m;
        }
        policy = std::atoi(argv[3]);
    }

    displayConfig(mix, policy, num_offloads);
    fillVectors(mix, num_pixels, input_buffers, output_buffers);

    switch (policy)
    {
    case 1:
        try
        {
            // Add CPUs to the universe of devices.
            resources.push_back(sycl::queue{sycl::cpu_selector_v});
        }
        catch (...)
        {
            std::cout << "Unable to create CPU queue\n";
        }
        invokeDS<ex::fixed_resource_policy<ex::sycl_backend>>(num_offloads, resources, num_pixels, input_buffers,
                                                              output_buffers);
        break;
    case 2:
        try
        {
            // Add GPUs to the universe of devices.
            resources.push_back(sycl::queue{sycl::gpu_selector_v});
        }
        catch (...)
        {
            std::cout << "Unable to create GPU queue\n";
        }
        invokeDS<ex::fixed_resource_policy<ex::sycl_backend>>(num_offloads, resources, num_pixels, input_buffers,
                                                              output_buffers);
        break;
    case 3:
        try
        {
            // Add CPUs and GPUs to the universe of devices.
            resources.push_back(sycl::queue{sycl::cpu_selector_v});
            resources.push_back(sycl::queue{sycl::gpu_selector_v});
        }
        catch (...)
        {
            std::cout << "Unable to create queues\n";
        }
        invokeDS<ex::round_robin_policy<ex::sycl_backend>>(num_offloads, resources, num_pixels, input_buffers,
                                                           output_buffers);
        break;
    case 4:
        try
        {
            resources.push_back(sycl::queue{sycl::gpu_selector_v});
            resources.push_back(sycl::queue{sycl::cpu_selector_v});
        }
        catch (...)
        {
            std::cout << "Unable to create queues\n";
        }
        invokeDS<ex::dynamic_load_policy<ex::sycl_backend>>(num_offloads, resources, num_pixels, input_buffers,
                                                            output_buffers);
        break;
    case 5:
        try
        {
            auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
            resources.push_back(sycl::queue{sycl::gpu_selector_v, prop_list});
            resources.push_back(sycl::queue{sycl::cpu_selector_v, prop_list});
        }
        catch (...)
        {
            std::cout << "Unable to create queues\n";
        }
        invokeDS<ex::auto_tune_policy<ex::sycl_backend, std::size_t>>(num_offloads, resources, num_pixels,
                                                                      input_buffers, output_buffers);
        break;
    default:
        std::cout << "Invalid policy." << std::endl;
        printUsage(argv[0]);
        return -1;
        break;
    }

    writeImages(output_buffers);

    return 0;
}
