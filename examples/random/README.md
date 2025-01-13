# Random number generator

This example demonstrates how to use scalar and vector random number generation using oneDPL.

| Optimized for                   | Description                                                                    |
|---------------------------------|--------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                           |
| Hardware                        | Skylake with GEN9 or newer                                                     |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL) |
| Time to complete                | At most 1 minute                                                               |

## License

This code example is licensed under [Apache License Version 2.0 with LLVM exceptions](https://github.com/uxlfoundation/oneDPL/blob/main/LICENSE.txt). Refer to the "[LICENSE](licensing/LICENSE.txt)" file for the full license text and copyright notice.

## Building the 'Random' Program for CPU and GPU

### On a Linux* System
Perform the following steps:

1. Source Intel&reg; oneAPI DPC++/C++ Compiler and oneDPL

2. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

3. Run the program:
```
    $ make run
```

4. Clean the program using:
```
    $ make clean
```

## Running the Program
### Example of Output

```
success for scalar generation
First 5 samples of minstd_rand with scalar generation
0.0174654
0.070213
0.250974
0.781962
0.10158

Last 5 samples of minstd_rand with scalar generation
0.496244
0.560782
0.537788
0.978197
0.163866

success for vector generation
First 5 samples of minstd_rand with vector generation
0.0174654
0.070213
0.250974
0.781962
0.10158

Last 5 samples of minstd_rand with vector generation
0.496244
0.560782
0.537788
0.978197
0.163866
```
