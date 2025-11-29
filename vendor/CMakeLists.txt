
include(FetchContent)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_ASSEMBLY_TESTS OFF)  # Resolves a warning about assembly tests
set(BENCHMARK_ENABLE_TESTING OFF)          # We don't need to build the benchmark tests
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(benchmark)


FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
# On Windows make gtest use the same (dynamic) CRT as the project
set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)
FetchContent_MakeAvailable(googletest)