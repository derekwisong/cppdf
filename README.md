# Prerequisites

- Intel Thread Building Blocks (TBB)
  - Arch: `sudo pacman -Sy intel-oneapi-tbb`
  - Fedora: `sudo dnf install tbb-devel` 

# Instructions

- Execute `./build.bash` to compile the project
- The following binaries will be produced:
  - `./build/src/benchmarks/benchmarks`
  - `./build/src/tests/tests`

# Example

Get the code, compile it, and run the benchmarks.

```bash
$ git clone git@github.com:derekwisong/cppdf.git
$ cd cppdf
$ ./build.bash
$ ./build/src/benchmarks/benchmarks
```
