add_executable(extra_buffer_halox HaloExchangeWithExtraBuffers.cpp codelets/HaloExchangeCommon.h)
add_executable(halox_approaches HaloRegionApproaches.cpp codelets/HaloExchangeCommon.h StructuredGridUtils.hpp GraphcoreUtils.hpp)

target_link_libraries(extra_buffer_halox
        poplar
        poputil
        popops
        )


target_link_libraries(halox_approaches
        poplar
        poputil
        popops
        )

configure_file(codelets/HaloExchangeCodelets.cpp codelets/HaloExchangeCodelets.cpp COPYONLY)
configure_file(codelets/HaloExchangeCommon.h codelets/HaloExchangeCommon.h COPYONLY)
configure_file(codelets/HaloRegionApproachesCodelets.cpp codelets/HaloRegionApproachesCodelets.cpp COPYONLY)
