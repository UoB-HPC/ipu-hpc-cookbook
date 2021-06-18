add_executable(particles ParticleShedding.cpp.cpp)

target_link_libraries(particles
        poplar
        poputil
        popops
        )

configure_file(codelets/ParticleSimCodelet.cpp codelets/ParticleSimCodelet.cpp COPYONLY)
configure_file(codelets/ParticleSimCodelet.p codelets/ParticleSimCodelet.cpp.cpp COPYONLY)
