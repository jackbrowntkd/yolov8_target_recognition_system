cmake_minimum_required(VERSION 3.16)
project(YOLOApp LANGUAGES NONE)

# Prefer local virtual environment Python when available.
if(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")
    set(PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")
else()
    find_program(PYTHON_EXECUTABLE NAMES python python3 REQUIRED)
endif()

message(STATUS "Using Python interpreter: ${PYTHON_EXECUTABLE}")
message(STATUS "Available targets: run_backend, run_frontend, run_all")

add_custom_target(run_backend
    COMMAND ${PYTHON_EXECUTABLE} -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Starting backend (FastAPI + Uvicorn)"
)

add_custom_target(run_frontend
    COMMAND ${PYTHON_EXECUTABLE} -m streamlit run client.py --server.address 0.0.0.0 --server.port 8501
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Starting frontend (Streamlit)"
)

if(WIN32)
    # Use Start-Process to launch each service in its own console window.
    add_custom_target(run_all
        COMMAND powershell -NoProfile -Command "Start-Process -FilePath '${PYTHON_EXECUTABLE}' -ArgumentList @('-m','uvicorn','server:app','--host','0.0.0.0','--port','8000','--reload') -WorkingDirectory '${CMAKE_SOURCE_DIR}'"
        COMMAND powershell -NoProfile -Command "Start-Process -FilePath '${PYTHON_EXECUTABLE}' -ArgumentList @('-m','streamlit','run','client.py','--server.address','0.0.0.0','--server.port','8501') -WorkingDirectory '${CMAKE_SOURCE_DIR}'"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Starting backend and frontend in parallel"
    )
else()
    add_custom_target(run_all
        COMMAND /bin/sh -c "${PYTHON_EXECUTABLE} -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload & ${PYTHON_EXECUTABLE} -m streamlit run client.py --server.address 0.0.0.0 --server.port 8501"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        COMMENT "Starting backend and frontend in parallel"
    )
endif()
