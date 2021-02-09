add_executable(skeleton SkeletonForIpu.cpp)

target_link_libraries(skeleton
        poplar
        poputil
        popops
        )

configure_file(codelets/SkeletonCodelets.cpp codelets/SkeletonCodelets.cpp COPYONLY)
