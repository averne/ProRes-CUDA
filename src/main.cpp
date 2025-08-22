/*
 * Copyright (c) 2025 averne <averne381@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cstdint>
#include <string>
#include <getopt.h>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

#include "util.hpp"
#include "decoder.hpp"

#define LAV_CHECK(expr) ({                                                      \
    if (auto _res_ = (expr); _res_ < 0) {                                       \
        char _tmpstr_[AV_ERROR_MAX_STRING_SIZE];                                \
        std::fprintf(stderr, STRING(expr) ": error %d (%s)\n", _res_,           \
                     av_make_error_string(_tmpstr_, sizeof(_tmpstr_), _res_));  \
        return _res_;                                                           \
    }                                                                           \
})

int bail(int argc, char **argv) {
    std::fprintf(stderr, "Usage: %s [--skip-idct|-i] [--skip-color|-c] [--skip-alpha|-a] "
                         "[--hash|-x] [--stdout|-o] [--path|-p path] [--frames|-f num_frames] "
                         "<video>\n", argv[0]);
    return 1;
}

int main(int argc, char **argv) {
    std::string o_path;
    bool skip_idct = false, skip_color = false, skip_alpha = false,
         o_hash    = false, o_stdout   = false, o_file     = false;
    int num_frames = INT_MAX;

    const struct option long_options[] = {
        {"skip-idct",  no_argument,       nullptr, 'i'},
        {"skip-color", no_argument,       nullptr, 'c'},
        {"skip-alpha", no_argument,       nullptr, 'a'},
        {"hash",       no_argument,       nullptr, 'x'},
        {"stdout",     no_argument,       nullptr, 'o'},
        {"path",       required_argument, nullptr, 'p'},
        {"frames",     required_argument, nullptr, 'f'},
        {"help",       no_argument,       nullptr, 'h'},
        {},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "icaxop:f:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i': skip_idct  = true; break;
            case 'c': skip_color = true; break;
            case 'a': skip_alpha = true; break;
            case 'x': o_hash     = true; break;
            case 'o': o_stdout   = true; break;
            case 'p': o_file     = true; o_path = optarg; break;
            case 'f': num_frames = std::atoi(optarg); break;
            case 'h':
            default:
                return bail(argc, argv);
        }
    }

    if (optind >= argc)
        return bail(argc, argv);

    std::fprintf(stderr, "Opening %s with %s-%s-%s\n", argv[optind],
                 LIBAVUTIL_IDENT, LIBAVFORMAT_IDENT, LIBAVCODEC_IDENT);

    AVFormatContext *fmt_ctx = nullptr;
    SCOPEGUARD([&fmt_ctx] { avformat_close_input(&fmt_ctx); });

    LAV_CHECK(avformat_open_input(&fmt_ctx, argv[optind], nullptr, nullptr));
    LAV_CHECK(avformat_find_stream_info(fmt_ctx, nullptr));

    int stream_idx;
    LAV_CHECK(stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0));

    auto *stream = fmt_ctx->streams[stream_idx];
    if (stream->codecpar->codec_id != AV_CODEC_ID_PRORES) {
        std::fprintf(stderr, "Unsupported codec %s\n", avcodec_get_name(stream->codecpar->codec_id));
        return 1;
    }

    auto pixfmt = static_cast<AVPixelFormat>(stream->codecpar->format);

    std::fprintf(stderr, "Video stream: idx %d, %s profile, %dx%d %s\n", stream_idx,
        avcodec_profile_name(stream->codecpar->codec_id, stream->codecpar->profile),
        stream->codecpar->width, stream->codecpar->height, av_get_pix_fmt_name(pixfmt));

    AVPacket *pkt = av_packet_alloc();
    SCOPEGUARD([&pkt] { av_packet_free(&pkt); });

    AVFrame *fr = av_frame_alloc();
    SCOPEGUARD([&fr] { av_frame_free(&fr); });
    fr->width = stream->codecpar->width, fr->height = stream->codecpar->height;

    auto sz = av_image_get_buffer_size(pixfmt, stream->codecpar->width, stream->codecpar->height, 1);

    auto *dat = av_malloc(sz);
    SCOPEGUARD([&dat] { av_free(dat); });

    LAV_CHECK(av_image_fill_arrays(fr->data, fr->linesize, static_cast<std::uint8_t *>(dat),
                                   pixfmt, stream->codecpar->width, stream->codecpar->height, 1));

    XXH64_state_t hash_state;
    if (o_hash)
        XXH64_reset(&hash_state, 0);

    FILE *fp = nullptr;
    SCOPEGUARD([&fp] { if (fp) std::fclose(fp); });
    if (o_file)
        fp = std::fopen(o_path.c_str(), "wb");

    auto decoder = CudaProresDecoder(stream->codecpar->bits_per_raw_sample, skip_idct, skip_color, skip_alpha);
    for (int i = 0; i < num_frames && av_read_frame(fmt_ctx, pkt) >= 0; ++i) {
        SCOPEGUARD([&pkt] { av_packet_unref(pkt); });

        if (pkt->stream_index != stream_idx)
            continue;

        std::fprintf(stderr, "Decoding packet %03d: size %#x\n", i, pkt->size);

        decoder.decode(pkt->buf, fr);

        for (int i = 0; i < 4; ++i) {
            if (fr->data[i]) {
                if (i <= 2 && skip_color) continue;
                if (i  > 2 && skip_alpha) continue;
                if (o_hash)   XXH64_update(&hash_state, fr->data[i], fr->linesize[i] * fr->height);
                if (o_stdout) std::fwrite(fr->data[i], fr->linesize[i], fr->height, stdout);
                if (o_file)   std::fwrite(fr->data[i], fr->linesize[i], fr->height, fp);
            }
        }
    }

    if (o_hash)
        std::printf("%016" PRIx64 "\n" , XXH64_digest(&hash_state));
}
