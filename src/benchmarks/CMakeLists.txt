add_executable(
    benchmarks
    main.cpp
    rng.cpp
    rng.h
 )

target_link_libraries(benchmarks PUBLIC dataframe benchmark)