#include <cstdint>
#include <array>
#include <vector>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include "util.hpp"

constexpr inline std::size_t ProresMbSize = 16;

enum class ProresProfile {
    Proxy    = 0,
    LT       = 1,
    Standard = 2,
    HQ       = 3,
    P4444    = 4,
    XQ       = 5,
};

enum class ProresChromaFormat {
    C422 = 2,
    C444 = 3,
};

enum class ProresInterlaceMode {
    Progressive   = 0,
    InterlacedTff = 1,
    InterlacedBff = 2,
};

enum class ProresAlphaType {
    None = 0,
    B8   = 1,
    B16  = 2,
};

using ProresQuantizationMatrix = std::uint8_t[8][8];

struct ProresFrame {
    ProresFrame(AVBufferRef *buf): buf(av_buffer_ref(buf)) { }
   ~ProresFrame() { av_buffer_unref(&this->buf); }

    AVBufferRef *buf;
    std::uint32_t picture_size;

    std::uint16_t width, height;
    std::uint16_t pic_width, pic_height;
    std::uint16_t mb_width, mb_height;
    std::uint16_t slice_width;
    std::uint8_t log2_slice_mb_width;

    ProresChromaFormat  chroma_fmt;
    ProresInterlaceMode interlace_mode;
    ProresAlphaType     alpha_type;

    bool first_field, bottom_field;

    ProresQuantizationMatrix luma_qmat, chroma_qmat;

    std::vector<std::uint16_t> slice_sizes;
};

class CudaProresDecoder {
    public:
        CudaProresDecoder(int depth): depth(depth) { };

        int decode(ProresFrame frame, AVFrame *dst);

    private:
        int parse_frame_header(ProresFrame &frame, util::Bytestream &bs);
        int parse_picture_header(ProresFrame &frame, util::Bytestream &bs);

    private:
        int depth;
};
