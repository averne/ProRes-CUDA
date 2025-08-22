# Prores-CUDA

Port of my [Vulkan ProRes decoder](https://github.com/averne/FFmpeg/tree/vk-proresdec) to CUDA, for easier debugging, profiling and testing.  
Dumps all frames of a ProRes video to separate YUV files.  
Supports all codec features:
- 4:2:2 and 4:4:4 subsampling
- 10 and 12-bit depth
- Alpha plane
- Interlacing

Disclaimer: not tested thorougly.

Depends on the FFmpeg libraries for demuxing.
