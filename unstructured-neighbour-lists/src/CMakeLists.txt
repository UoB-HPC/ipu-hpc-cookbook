add_executable(unstructured Unstructured.cpp)

target_link_libraries(unstructured
        poplar
        poputil
        popops
        )

configure_file(codelets/UnstructuredCodelets.cpp codelets/UnstructuredCodelets.cpp COPYONLY)
target_compile_features(unstructured PRIVATE cxx_std_17)