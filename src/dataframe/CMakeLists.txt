
add_library(
    dataframe
    STATIC
    dataframe.cpp
    series.cpp
)

target_include_directories(
    dataframe
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link against Threads library (pthread or equivalent)
find_package(Threads REQUIRED)
target_link_libraries(dataframe
    PRIVATE
        Threads::Threads
)

# Enable Intel TBB support if available
# See https://github.com/justusc/FindTBB 
#     https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "TBB found. Enabling TBB support.")
    target_link_libraries(dataframe
        PUBLIC
            TBB::tbb
    )
    target_compile_definitions(dataframe
        PUBLIC
            USE_TBB=1
    )
else()
    message(STATUS "TBB not found. Building without TBB support.")
endif()
