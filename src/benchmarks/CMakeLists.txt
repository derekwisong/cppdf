add_executable(
    benchmarks
    main.cpp
)

target_link_libraries(benchmarks PUBLIC dataframe benchmark)