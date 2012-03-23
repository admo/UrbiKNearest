set(URBI_ROOT_DIR "/usr" CACHE PATH "Urbi root directory.")

find_package (OpenCV REQUIRED)
find_package (Urbi REQUIRED)
find_package (Boost REQUIRED serialization)

link_directories (${BOOST_LIBRARYDIR})

include_directories (${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${URBI_INCLUDE_DIRS})

if (WIN32)
  add_definitions( -DBOOST_ALL_DYN_LINK )
endif (WIN32)

add_library (uknearest MODULE uknearest.cpp)

target_link_libraries (uknearest ${OpenCV_LIBS} ${URBI_LIBRARIES} ${Boost_LIBRARIES})

set_target_properties (uknearest PROPERTIES
  PREFIX "")
  
install (TARGETS uknearest DESTINATION lib/gostai/uobjects COMPONENT libraries)
