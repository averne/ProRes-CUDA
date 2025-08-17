project('ProRes-CUDA', ['cpp', 'cuda'],
    default_options: ['cpp_std=c++20']
)

if get_option('buildtype') == 'debug'
    add_global_arguments('-DDEBUG', language: 'cpp')
endif

executable('prores-cuda', 'src/main.cpp', 'src/decoder.cu',
    dependencies: [
        dependency('libavutil',   required: true),
        dependency('libavformat', required: true),
        dependency('libavcodec',  required: true),
    ],
)
