# wyoming-whisper-cpp

Simple STT server using wyoming protocol and whisper.cpp.

"""
This code is presented as-is, without any warranties or guarantees of functionality. 
It serves as a simple implementation that can surely be improved or optimized 
based on specific requirements or use cases. Users are encouraged to review, 
test, and modify the code as needed to suit their needs.
"""

In the project ther's the code to build a docker image using vulkan acceleration for devices that use the mesa drivers. To use other frameworks to accelerate the workload, follow instructions on the whisper.cpp github to compile whisper.cpp and copy the necessary files. New dockerfiles for different frameworks are welcome. 

## Project Components

- **simple-server**: lightweight C++ server based on the stream example provided by whisper.cpp.
- **wyoming_whisper_cpp**: python module implementing a Wyoming protocol server.


## Build Docker image

Build docker image:

```shell
docker build -f dockerfiles/ubuntu-vulkan-mesa -t wyoming-whisper-cpp:vulkan .
```

If you don't want to use docker, just follow all the command used in the dockerfile 

## Run with docker-compose

```shell
docker-compose up -d
```
