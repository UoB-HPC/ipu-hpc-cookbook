add_executable(unstructured Unstructured.cpp)

target_link_libraries(unstructured
        poplar
        poputil
        popops
        )

configure_file(codelets/UnstructuredCodelets.cpp codelets/UnstructuredCodelets.cpp COPYONLY)
