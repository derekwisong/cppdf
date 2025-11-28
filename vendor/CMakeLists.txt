
include(FetchContent)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(benchmark)
