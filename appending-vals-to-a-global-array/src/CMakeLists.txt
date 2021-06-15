add_executable(appending_to_global_arr AppendingToGlobalArray.cpp)

target_link_libraries(appending_to_global_arr
        poplar
        poputil
        popops
        )

configure_file(codelets/AppendingValsToGlobal.cpp codelets/AppendingValsToGlobal.cpp COPYONLY)
