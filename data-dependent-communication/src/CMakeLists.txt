add_executable(particles ParticleShedding.cpp)

target_link_libraries(particles
        poplar
        poputil
        popops
        )

configure_file(codelets/ParticleSimCodelet.cpp codelets/ParticleSimCodelet.cpp COPYONLY)
configure_file(codelets/ParticleSimCodelet.cpp codelets/ParticleSimCodelet.cpp COPYONLY)
