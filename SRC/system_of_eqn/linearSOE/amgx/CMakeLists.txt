target_sources(OPS_SysOfEqn
    PRIVATE
        AmgXGenLinSOE.cpp
        AmgXGenLinSolver.cpp

    PUBLIC
        AmgXGenLinSOE.h
        AmgXGenLinSolver.h

)

target_include_directories(OPS_SysOfEqn PUBLIC ${CMAKE_CURRENT_LIST_DIR})