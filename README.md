# Prores-CUDA

Port of my [Vulkan ProRes decoder](https://github.com/averne/FFmpeg/tree/vk-proresdec) to CUDA, for easier debugging, profiling and testing.

Supports all codec features:
- 4:2:2 and 4:4:4 subsampling
- 10 and 12-bit depth
- Alpha plane
- Interlacing (outputs each field in a separate full-height frame)

Options:
- Specify number of frames to decode
- Skip color decoding, alpha decoding or IDCT
- Output to a raw YUV file, stdout or a hash function (xxhash64) 

Disclaimer: not tested thorougly.

Depends on the FFmpeg libraries for demuxing.
