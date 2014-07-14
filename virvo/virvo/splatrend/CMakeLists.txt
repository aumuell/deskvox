add_definitions(-DVIRVO_EXPORTS)


find_package(Boost REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Pthreads REQUIRED)
find_package(X11 REQUIRED)

deskvox_use_package(Boost)
deskvox_use_package(GLEW)
deskvox_use_package(OpenCL)
deskvox_use_package(Pthreads)
deskvox_use_package(X11)

deskvox_link_libraries(virvo)

set(SPLATREND_HEADERS
  ../splatrend.h
)

set(SPLATREND_SOURCES
  ../splatrend.cpp
)

add_definitions(-DHAVE_CONFIG_H)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

deskvox_add_library(splatrend
  ${SPLATREND_HEADERS}
  ${SPLATREND_SOURCES}
)


