# Dot product

This example contains the oneDPL-based implementation of dot product based on `std::transform_reduce`.

| Optimized for                   | Description                                                                                    |
|---------------------------------|------------------------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                                           |
| Hardware                        | Skylake or newer                                                                               |
| Software                        | Intel&reg; oneAPI DPC++ Library (oneDPL); Intel&reg; oneAPI Threading Building Blocks (oneTBB) |
| Time to complete                | At most 1 minute                                                                               |

## License

This code example is licensed under [Apache License Version 2.0 with LLVM exceptions](https://github.com/uxlfoundation/oneDPL/blob/main/LICENSE.txt). Refer to the "[LICENSE](licensing/LICENSE.txt)" file for the full license text and copyright notice.

## Building the 'Dot product' Program

### On a Linux* System
Perform the following steps:

1. Source oneDPL and oneTBB

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
success
The dot product is: 2.49872e+06
```
