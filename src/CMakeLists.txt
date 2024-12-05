file(GLOB_RECURSE SOURCES
    "**/*.cpp"
    "*.cpp"
)

add_library(RoadRecognition ${SOURCES} )

target_include_directories(RoadRecognition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})