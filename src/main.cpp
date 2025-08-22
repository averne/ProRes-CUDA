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

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include "util.hpp"
#include "decoder.hpp"

#define LAV_CHECK(expr) ({                                                      \
    if (auto _res_ = (expr); _res_ < 0) {                                       \
        char _tmpstr_[AV_ERROR_MAX_STRING_SIZE];                                \
        std::printf(STRING(expr) ": error %d (%s)\n", _res_,                    \
                    av_make_error_string(_tmpstr_, sizeof(_tmpstr_), _res_));   \
        return _res_;                                                           \
    }                                                                           \
})

int main(int argc, char **argv) {
    std::printf("Starting now, %s-%s-%s\n", LIBAVUTIL_IDENT, LIBAVFORMAT_IDENT, LIBAVCODEC_IDENT);

    if (argc < 2) {
        std::printf("Usage: %s input\n", argv[0]);
        return 1;
    }

    AVFormatContext *fmt_ctx = nullptr;
    SCOPEGUARD([&fmt_ctx] { avformat_close_input(&fmt_ctx); });

    LAV_CHECK(avformat_open_input(&fmt_ctx, argv[1], nullptr, nullptr));
    LAV_CHECK(avformat_find_stream_info(fmt_ctx, nullptr));

    int stream_idx;
    LAV_CHECK(stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0));

    auto *stream = fmt_ctx->streams[stream_idx];
    if (stream->codecpar->codec_id != AV_CODEC_ID_PRORES) {
        std::printf("Unsupported codec %s\n", avcodec_get_name(stream->codecpar->codec_id));
        return 1;
    }

    auto pixfmt = static_cast<AVPixelFormat>(stream->codecpar->format);

    std::printf("Video stream: idx %d, %s profile, %dx%d %s\n", stream_idx,
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

    std::string path(0x20, '\0');
    auto decoder = CudaProresDecoder(stream->codecpar->bits_per_raw_sample);
    for (int i = 0; av_read_frame(fmt_ctx, pkt) >= 0; ++i) {
        SCOPEGUARD([&pkt] { av_packet_unref(pkt); });

        if (pkt->stream_index != stream_idx)
            continue;

        std::printf("Decoding packet %03d: size %#x\n", i, pkt->size);

        decoder.decode(pkt->buf, fr);

        std::snprintf(path.data(), path.size(), "frame-%d.yuv", i);

        auto *fp = std::fopen(path.data(), "wb");
        SCOPEGUARD([&fp] { std::fclose(fp); });

        for (int i = 0; i < 4; ++i) {
            if (fr->data[i])
                std::fwrite(fr->data[i], fr->linesize[i], fr->height, fp);
        }
    }
}

