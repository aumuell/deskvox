if(NOT DESKVOX_BUILD_TOOLS)
  return()
endif()

if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DVIRVO_STATIC)
endif()

deskvox_link_libraries(virvo)

ADD_SUBDIRECTORY(vconv)
if(DESKVOX_USE_ASIO)
  ADD_SUBDIRECTORY(vserver_asio)
else()
  ADD_SUBDIRECTORY(vserver)
endif()
ADD_SUBDIRECTORY(vview)
