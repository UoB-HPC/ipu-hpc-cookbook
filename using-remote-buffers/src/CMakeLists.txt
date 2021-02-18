add_executable(remote-buffers RemoteBuffers.cpp)
configure_file(codelets/RemoteBuffers.cpp codelets/RemoteBuffers.cpp COPYONLY)

target_link_libraries(remote-buffers
        poplar
        poputil
        popops
        )

target_include_directories(remote-buffers PUBLIC include)