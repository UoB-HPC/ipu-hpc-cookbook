add_executable(extra-buffer-halox HaloExchangeWithExtraBuffers.cpp)
add_executable(halox-appraches HaloExchangeWithExtraBuffers.cpp)

target_link_libraries(skeleton
        poplar
        poputil
        popops
        )

configure_file(codelets/HaloExchangeCodelets.cpp codelets/HaloExchangeCodelets.cpp COPYONLY)
configure_file(codelets/HaloExchangeCommon.h codelets/HaloExchangeCommon.h COPYONLY)
configure_file(codelets/HaloRegionApproachesCodelets.cpp codelets/HaloExchangeCommon.h COPYONLY)
