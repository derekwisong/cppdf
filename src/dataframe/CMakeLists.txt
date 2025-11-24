
add_library(
    dataframe
    STATIC
    dataframe.cpp
    series.cpp
)

target_link_libraries(
    dataframe
    PUBLIC
    pthread
    tbb
)

target_include_directories(
    dataframe
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)