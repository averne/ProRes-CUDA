#include <bit>
#include <numeric>

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/pixdesc.h>
}

#include "common.cuh"

#include "decoder.hpp"

struct KernelParams {
    CUdeviceptr slice_data;
    std::uint32_t bitstream_size;

    std::uint16_t width;
    std::uint16_t height;
    std::uint16_t mb_width;
    std::uint16_t mb_height;
    std::uint16_t slice_width;
    std::uint16_t slice_height;
    std::uint8_t  log2_chroma_w;
    std::uint8_t  depth;
    std::uint8_t  alpha_info;
    std::uint8_t  bottom_field;

    std::uint8_t  qmat_luma  [8][8];
    std::uint8_t  qmat_chroma[8][8];
};

struct SliceContext {
    std::uint16_t mb_x;
    std::uint16_t mb_y;
    std::uint8_t  mb_count;
};

__constant__ KernelParams params;

template <bool interlaced> __device__
static inline uint get_px(cudaSurfaceObject_t src, int2 pos) {
    if constexpr (!interlaced)
        return surf2Dread<std::uint16_t>(src, pos.x * sizeof(std::uint16_t), pos.y);
    else
        return surf2Dread<std::uint16_t>(src, pos.x * sizeof(std::uint16_t), (pos.y << 1) + params.bottom_field);
}

template <bool interlaced> __device__
static inline void put_px(cudaSurfaceObject_t dst, int2 pos, uint v) {
    if constexpr (!interlaced)
        surf2Dwrite<std::uint16_t>(v, dst, pos.x * sizeof(std::uint16_t), pos.y);
    else
        surf2Dwrite<std::uint16_t>(v, dst, pos.x * sizeof(std::uint16_t), (pos.y << 1) + params.bottom_field);
}

/* 7.5.3 Pixel Arrangement */
__device__
int2 pos_to_block(uint pos, uint luma)
{
    return int2((pos & -luma - 2) + luma >> 1, pos >> luma & 1) << 3;
}

/* 7.1.1.2 Signed Golomb Combination Codes */
__device__
uint to_signed(uint x)
{
    return (x >> 1) ^ -(x & 1);
}

/* 7.1.1.1 Golomb Combination Codes */
__device__
uint decode_codeword(util::Bitstream &gb, std::uint16_t codebook)
{
    uint last_rice_q = (codebook >> 0) & 15,
         krice       = (codebook >> 4) & 15,
         kexp        = (codebook >> 8);

    uint q = 31 - findMSB(gb.show_bits(32));
    if (q <= last_rice_q) {
        /* Golomb-Rice encoding */
        return (gb.get_bits(krice + q + 1) & ~(1 << krice)) + (q << krice);
    } else {
        /* exp-Golomb encoding */
        return gb.get_bits((q << 1) + kexp - last_rice_q) - (1 << kexp) + ((last_rice_q + 1) << krice);
    }
}

template <bool interlaced> __device__
void decode_comp(cudaSurfaceObject_t dst, util::Bitstream gb, SliceContext ctx, uint qscale)
{
    uint is_luma = uint(blockIdx.z == 0);
    uint chroma_shift = bool(is_luma) ? 0 : params.log2_chroma_w;

    uint num_blocks = ctx.mb_count << (2 - chroma_shift);
    int2 base_pos = int2(ctx.mb_x << (4 - chroma_shift), ctx.mb_y << 4);

    /* 7.1.1.3 DC Coefficients */
    {
        /* First coeff */
        uint c = to_signed(decode_codeword(gb, U16(0x650)));
        put_px<interlaced>(dst, base_pos, c * qscale & 0xffff);

        /**
         * Table 9, encoded as (last_rice_q << 0) | (krice or kexp << 4) | ((kexp or kexp + 1) << 8)
         * According to the SMPTE document, abs(prev_dc_diff) should be used
         * to index the table, duplicating the entries removes the abs operation
         */
        const uint16_t dc_codebook[] = { U16(0x100),
                                         U16(0x210), U16(0x210),
                                         U16(0x321), U16(0x321),
                                         U16(0x430), U16(0x430), };

        uint cw = 5, prev_dc_diff = 0;
        for (int i = 1; i < num_blocks; ++i) {
            cw = decode_codeword(gb, dc_codebook[min(cw, 6)]);

            int s = int(prev_dc_diff) >> 31;
            c += prev_dc_diff = (to_signed(cw) ^ s) - s;

            put_px<interlaced>(dst, base_pos + pos_to_block(i, is_luma), c * qscale & 0xffff);
        }
    }

    /* 7.1.1.4 AC Coefficients */
    {
        /* Table 10 */
        const uint16_t ac_run_codebook  [] = { U16(0x102), U16(0x102), U16(0x101), U16(0x101),
                                               U16(0x100), U16(0x211), U16(0x211), U16(0x211),
                                               U16(0x211), U16(0x210), U16(0x210), U16(0x210),
                                               U16(0x210), U16(0x210), U16(0x210), U16(0x320), };

        /* Table 11 */
        const uint16_t ac_level_codebook[] = { U16(0x202), U16(0x101), U16(0x102), U16(0x100),
                                               U16(0x210), U16(0x210), U16(0x210), U16(0x210),
                                               U16(0x320) };

        /* Figure 4, encoded as (x << 0) | (y << 4) */
        const uint8_t scan_tbl_prog[] = {
            U8(0x00), U8(0x01), U8(0x10), U8(0x11), U8(0x02), U8(0x03), U8(0x12), U8(0x13),
            U8(0x20), U8(0x21), U8(0x30), U8(0x31), U8(0x22), U8(0x23), U8(0x32), U8(0x33),
            U8(0x04), U8(0x05), U8(0x14), U8(0x24), U8(0x15), U8(0x06), U8(0x07), U8(0x16),
            U8(0x25), U8(0x34), U8(0x35), U8(0x26), U8(0x17), U8(0x27), U8(0x36), U8(0x37),
            U8(0x40), U8(0x41), U8(0x50), U8(0x60), U8(0x51), U8(0x42), U8(0x43), U8(0x52),
            U8(0x61), U8(0x70), U8(0x71), U8(0x62), U8(0x53), U8(0x44), U8(0x45), U8(0x54),
            U8(0x63), U8(0x72), U8(0x73), U8(0x64), U8(0x55), U8(0x46), U8(0x47), U8(0x56),
            U8(0x65), U8(0x74), U8(0x75), U8(0x66), U8(0x57), U8(0x67), U8(0x76), U8(0x77),
        };

        /* Figure 5 */
        const uint8_t scan_tbl_itld[] = {
            U8(0x00), U8(0x10), U8(0x01), U8(0x11), U8(0x20), U8(0x30), U8(0x21), U8(0x31),
            U8(0x02), U8(0x12), U8(0x03), U8(0x13), U8(0x22), U8(0x32), U8(0x23), U8(0x33),
            U8(0x40), U8(0x50), U8(0x41), U8(0x42), U8(0x51), U8(0x60), U8(0x70), U8(0x61),
            U8(0x52), U8(0x43), U8(0x53), U8(0x62), U8(0x71), U8(0x72), U8(0x63), U8(0x73),
            U8(0x04), U8(0x14), U8(0x05), U8(0x06), U8(0x15), U8(0x24), U8(0x34), U8(0x25),
            U8(0x16), U8(0x07), U8(0x17), U8(0x26), U8(0x35), U8(0x44), U8(0x54), U8(0x45),
            U8(0x36), U8(0x27), U8(0x37), U8(0x46), U8(0x55), U8(0x64), U8(0x74), U8(0x65),
            U8(0x56), U8(0x47), U8(0x57), U8(0x66), U8(0x75), U8(0x76), U8(0x67), U8(0x77),
        };

        auto scan_tbl = !interlaced ? scan_tbl_prog : scan_tbl_itld;
        uint block_mask  = num_blocks - 1;
        uint block_shift = findLSB(num_blocks);

        uint pos = num_blocks - 1, run = 4, level = 1, s;
        while (pos < (num_blocks << 6) - 1) {
            uint left = gb.left_bits();
            if (left <= 0 || (left < 32 && gb.show_bits(left) == 0))
                break;

            run   = decode_codeword(gb, ac_run_codebook  [min(run,   15)]);
            level = decode_codeword(gb, ac_level_codebook[min(level, 8 )]);
            s     = gb.get_bits(1);

            pos += run + 1;

            uint bidx = pos & block_mask, scan = scan_tbl[pos >> block_shift];
            int2 spos = pos_to_block(bidx, is_luma);
            int2 bpos = int2(scan & 0xf, scan >> 4);

            uint c = ((level + 1) ^ -s) + s;
            put_px<interlaced>(dst, base_pos + spos + bpos, c * qscale & 0xffff);
        }
    }
}

template <bool interlaced> __global__ __launch_bounds__(64)
void kern_color_vld(cudaSurfaceObject_t *dst, const std::uint32_t *slice_offsets, const SliceContext *slice_ctxs)
{
    auto gid = blockIdx * blockDim + threadIdx;
    if (gid.x >= params.slice_width || gid.y >= params.slice_height)
        return;

    uint slice_idx = gid.y * params.slice_width + gid.x;
    uint slice_off  = slice_offsets[slice_idx],
         slice_size = slice_offsets[slice_idx + 1] - slice_off;

    auto *bs = reinterpret_cast<const std::uint8_t *>(params.slice_data + slice_off);

    /* Decode slice header */
    uint hdr_size = 0, y_size = 0, u_size = 0, v_size = 0, qscale = 0;
    hdr_size = bs[0] >> 3;

    /* Table 15 */
    uint qidx = clamp(bs[1], 1, 224);
    qscale = qidx > 128 ? (qidx - 96) << 2 : qidx;

    y_size = (uint(bs[2]) << 8) | bs[3];
    u_size = (uint(bs[4]) << 8) | bs[5];

    /**
     * The alpha_info field can be 0 even when an alpha plane is present,
     * if skip_alpha is enabled, so use the header size instead.
     */
    if (hdr_size > 6)
        v_size = (uint(bs[6]) << 8) | bs[7];
    else
        v_size = slice_size - hdr_size - y_size - u_size;

    util::Bitstream gb;
    switch (gid.z) {
        case 0:
            gb.init(bs + hdr_size, y_size);
            break;
        case 1:
            gb.init(bs + hdr_size + y_size, u_size);
            break;
        case 2:
            gb.init(bs + hdr_size + y_size + u_size, v_size);
            break;
    }

    /**
     * Support for the grayscale "extension" in the prores_aw encoder.
     * According to the spec, entropy coded data should never be empty,
     * and instead contain at least the DC coefficients.
     * This avoids undefined behavior.
     */
    if (gb.left_bits() == 0)
        return;

    /* Entropy decoding, inverse scanning, first part of inverse quantization */
    decode_comp<interlaced>(dst[gid.z], gb, slice_ctxs[slice_idx], qscale);
}

/* 7.1.2 Scanned Alpha */
template <bool interlaced> __device__
void decode_alpha(cudaSurfaceObject_t dst, util::Bitstream gb, SliceContext ctx) {
    auto gid = blockIdx * blockDim + threadIdx;

    int2 base_pos = int2(ctx.mb_x, ctx.mb_y) << 4;
    uint block_shift = findMSB(ctx.mb_count) + 4, block_mask = (1 << block_shift) - 1;

    uint mask = (1 << (4 << params.alpha_info)) - 1;
    uint num_values = (ctx.mb_count << 4) * min(params.height - (gid.y << 4), 16);

    uint num_cw_bits  = params.alpha_info == 1 ? 5 : 8,
         num_flc_bits = params.alpha_info == 1 ? 9 : 17;

    uint alpha_rescale_lshift = params.alpha_info == 1 ? params.depth - 8 : 16,
         alpha_rescale_rshift = 16 - params.depth;

    uint alpha = -1u;
    for (uint pos = 0; pos < num_values;) {
        uint diff, run;

        /* Decode run value */
        {
            uint bits = gb.show_bits(num_cw_bits), q = num_cw_bits - 1 - findMSB(bits);

            /* Tables 13/14 */
            if (q != 0) {
                uint m = (bits >> 1) + 1, s = bits & 1;
                diff = (m ^ -s) + s;
                gb.skip_bits(num_cw_bits);
            } else {
                diff = gb.get_bits(num_flc_bits);
            }

            alpha = alpha + diff & mask;
        }

        /* Decode run length */
        {
            uint bits = gb.show_bits(5), q = 4 - findMSB(bits);

            /* Table 12 */
            if (q == 0) {
                run = 1;
                gb.skip_bits(1);
            } else if (q <= 4) {
                run = bits + 1;
                gb.skip_bits(5);
            } else {
                run = gb.get_bits(16) + 1;
            }

            run = min(run, num_values - pos);
        }

        /**
         * FFmpeg doesn't support color and alpha with different precision,
         * so we need to rescale to the color range.
         */
        uint val = (alpha << alpha_rescale_lshift) | (alpha >> alpha_rescale_rshift);
        for (uint end = pos + run; pos < end; ++pos)
            put_px<interlaced>(dst, base_pos + int2(pos & block_mask, pos >> block_shift), val & 0xffff);
    }
}

template <bool interlaced> __global__ __launch_bounds__(64)
void kern_alpha_vld(cudaSurfaceObject_t *dst, const std::uint32_t *slice_offsets, const SliceContext *slice_ctxs) {
    auto gid = blockIdx * blockDim + threadIdx;
    if (gid.x >= params.slice_width || gid.y >= params.slice_height)
        return;

    uint slice_idx = gid.y * params.slice_width + gid.x;
    uint slice_off  = slice_offsets[slice_idx],
         slice_size = slice_offsets[slice_idx + 1] - slice_off;

    /* Decode slice header */
    auto *bs = reinterpret_cast<const std::uint8_t *>(params.slice_data + slice_off);
    uint skip_size = (bs[0] >> 3) +
        ((uint(bs[2]) << 8) | bs[3]) +
        ((uint(bs[4]) << 8) | bs[5]) +
        ((uint(bs[6]) << 8) | bs[7]);

    util::Bitstream gb;
    gb.init(bs + skip_size, slice_size - skip_size);

    decode_alpha<interlaced>(dst[3], gb, slice_ctxs[slice_idx]);
}

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh
#define C_a 1.387039845322148f     //!< a = (2^0.5) * cos(    pi / 16);
#define C_b 1.306562964876377f     //!< b = (2^0.5) * cos(    pi /  8);
#define C_c 1.175875602419359f     //!< c = (2^0.5) * cos(3 * pi / 16);
#define C_d 0.785694958387102f     //!< d = (2^0.5) * cos(5 * pi / 16);
#define C_e 0.541196100146197f     //!< e = (2^0.5) * cos(3 * pi /  8);
#define C_f 0.275899379282943f     //!< f = (2^0.5) * cos(7 * pi / 16);
#define C_norm 0.3535533905932737f //!< 1 / (8^0.5)

/* 7.4 Inverse Transform */
__device__
void idct(float blocks[8][72], uint block, uint offset, uint stride) {
    float Vect0 = blocks[block][0*stride + offset];
    float Vect1 = blocks[block][1*stride + offset];
    float Vect2 = blocks[block][2*stride + offset];
    float Vect3 = blocks[block][3*stride + offset];
    float Vect4 = blocks[block][4*stride + offset];
    float Vect5 = blocks[block][5*stride + offset];
    float Vect6 = blocks[block][6*stride + offset];
    float Vect7 = blocks[block][7*stride + offset];

    float Y04P   = Vect0 + Vect4;
    float Y2b6eP = C_b * Vect2 + C_e * Vect6;

    float Y04P2b6ePP   = Y04P + Y2b6eP;
    float Y04P2b6ePM   = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * Vect7 + C_a * Vect1 + C_c * Vect3 + C_d * Vect5;
    float Y7a1fM3d5cMP = C_a * Vect7 - C_f * Vect1 + C_d * Vect3 - C_c * Vect5;

    float Y04M   = Vect0 - Vect4;
    float Y2e6bM = C_e * Vect2 - C_b * Vect6;

    float Y04M2e6bMP   = Y04M + Y2e6bM;
    float Y04M2e6bMM   = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * Vect1 - C_d * Vect7 - C_f * Vect3 - C_a * Vect5;
    float Y1d7cP3a5fMM = C_d * Vect1 + C_c * Vect7 - C_a * Vect3 + C_f * Vect5;

    blocks[block][0*stride + offset] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    blocks[block][7*stride + offset] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    blocks[block][4*stride + offset] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    blocks[block][3*stride + offset] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    blocks[block][1*stride + offset] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    blocks[block][5*stride + offset] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    blocks[block][2*stride + offset] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    blocks[block][6*stride + offset] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

template <bool interlaced> __global__ __launch_bounds__(64)
void kern_idct(cudaSurfaceObject_t *surf) {
    /* Two macroblocks, padded to avoid bank conflicts */
    __shared__ float blocks[4*2][8*(8+1)];

    auto gid = blockIdx * blockDim + threadIdx, lid = threadIdx;
    uint comp = gid.z, block = (lid.y << 2) | (lid.x >> 3), idx = lid.x & 0x7;
    uint chroma_shift = comp != 0 ? params.log2_chroma_w : 0;
    bool act = gid.x < params.mb_width << (4 - chroma_shift);

    if (act) {
        for (uint i = 0; i < 8; ++i) {
            int v = sign_extend(int(get_px<interlaced>(surf[comp], int2(gid.x, (gid.y << 3) | i))), 16);
            int w = comp == 0 ? params.qmat_luma[i][idx] : params.qmat_chroma[i][idx];
            blocks[block][i * 9 + idx] = float(v * w);
        }
    }

    /* Row-wise iDCT */
    __syncthreads();
    idct(blocks, block, idx * 9, 1);

    /* Column-wise iDCT */
    __syncthreads();
    idct(blocks, block, idx, 9);

    float fact = 1.0f / (1 << (12 - params.depth)), off = 1 << (params.depth - 1);
    int maxv = (1 << params.depth) - 1;

    /* 7.5.1 Color Component Samples. Rescale, clamp and write back to global memory */
    __syncthreads();
    if (act) {
        for (uint i = 0; i < 8; ++i) {
            float v = blocks[block][i * 9 + idx] * fact + off;
            put_px<interlaced>(surf[comp], int2(gid.x, (gid.y << 3) | i), clamp(int(v), 0, maxv));
        }
    }
}

int CudaProresDecoder::parse_frame_header(ProresFrame &frame, util::Bytestream &bs) {
    // 5.1.1 Frame Header Syntax
    auto hdr_size = bs.get_be<std::uint16_t>();
    if (bs.left() < static_cast<std::size_t>(hdr_size))
        return -1;

    // reserved, bitstream_version, encoder_identifier
    bs.skip(6);

    frame.width = bs.get_be<std::uint16_t>(), frame.height = bs.get_be<std::uint16_t>();

    auto flags = bs.get_be<std::uint8_t>();
    frame.chroma_fmt     = static_cast<ProresChromaFormat >(flags >> 6 & util::mask(2));
    frame.interlace_mode = static_cast<ProresInterlaceMode>(flags >> 2 & util::mask(2));

    // aspect_ratio_information, frame_rate_code, color_primaries, transfer_characteristic, matrix_coefficients
    bs.skip(4);

    frame.alpha_type = static_cast<ProresAlphaType>(bs.get_be<std::uint8_t>() & util::mask(4));

    auto qmat_flags = bs.get_be<std::uint16_t>();

    if (qmat_flags >> 0 & util::mask(1))
        bs.read_into(frame.luma_qmat[0], sizeof(frame.luma_qmat) * sizeof(std::uint8_t));
    else
        std::fill_n(frame.luma_qmat [0], sizeof(frame.luma_qmat) * sizeof(std::uint8_t), 4);

    if (qmat_flags >> 1 & util::mask(1))
        bs.read_into(frame.chroma_qmat[0], sizeof(frame.chroma_qmat) * sizeof(std::uint8_t));
    else
        std::copy_n(frame.luma_qmat   [0], sizeof(frame.luma_qmat  ) * sizeof(std::uint8_t), frame.chroma_qmat[0]);

    frame.bottom_field = frame.first_field ^ (frame.interlace_mode == ProresInterlaceMode::InterlacedTff);

    switch (frame.interlace_mode) {
        case ProresInterlaceMode::Progressive:
            frame.pic_width  = frame.width;
            frame.pic_height = frame.height;
            break;
        case ProresInterlaceMode::InterlacedTff:
        case ProresInterlaceMode::InterlacedBff:
            frame.pic_width  = frame.width;
            frame.pic_height = (frame.height + !frame.bottom_field) / 2;
            break;
    }

    frame.mb_width  = util::align_up(frame.pic_width,  ProresMbSize) / ProresMbSize;
    frame.mb_height = util::align_up(frame.pic_height, ProresMbSize) / ProresMbSize;

    return 0;
}

int CudaProresDecoder::parse_picture_header(ProresFrame &frame, util::Bytestream &bs) {
    // 5.2.1 Picture Header Syntax
    auto hdr_size = bs.get_be<std::uint8_t>() >> 3;
    if (bs.left() < static_cast<std::size_t>(hdr_size))
        return -1;

    frame.picture_size = bs.get_be<std::uint32_t>();

    // deprecated_number_of_slices
    bs.skip(2);

    frame.log2_slice_mb_width = bs.get_be<std::uint8_t>() >> 4;

    frame.slice_width = (frame.mb_width >> frame.log2_slice_mb_width) +
                        std::popcount(static_cast<std::uint16_t>(frame.mb_width & util::mask(frame.log2_slice_mb_width)));

    frame.slice_sizes.resize(frame.slice_width * frame.mb_height);
    std::generate_n(frame.slice_sizes.data(), frame.slice_sizes.size(), [&bs] { return bs.get_be<std::uint16_t>(); });

    return 0;
}

int CudaProresDecoder::decode(ProresFrame frame, AVFrame *dst) {
    auto bs = util::Bytestream(frame.buf->data, frame.buf->size);

    // 5.1 Frame Syntax
    auto frame_size = bs.get_be<std::uint32_t>(), frame_id = bs.get<std::uint32_t>();
    if (static_cast<std::size_t>(frame.buf->size) < frame_size || frame_id != util::FourCC('i','c','p','f'))
        return -1;

    frame.first_field = true;
    if (auto rc = this->parse_frame_header(frame, bs); rc < 0)
        return rc;

    if (auto rc = this->parse_picture_header(frame, bs); rc < 0)
        return rc;

    std::vector<std::uint32_t> slice_offsets_cpu(frame.slice_sizes.size() + 1, 0);
    std::inclusive_scan(frame.slice_sizes.begin(), frame.slice_sizes.end(), slice_offsets_cpu.begin() + 1,
                        std::plus(), std::uint32_t()); // Explicitly use 32-bit accumulator to avoid overflow

    int mb_x = 0, slice_mb_width = 1 << frame.log2_slice_mb_width;
    std::vector<SliceContext> slice_ctxs_cpu(frame.slice_sizes.size());
    for (std::size_t i = 0; i < frame.slice_sizes.size(); ++i) {
        auto &slice_ctx = slice_ctxs_cpu[i];

        while (frame.mb_width - mb_x < slice_mb_width)
            slice_mb_width >>= 1;

        slice_ctx.mb_x = mb_x;
        slice_ctx.mb_y = static_cast<std::uint16_t>(i / frame.slice_width);
        slice_ctx.mb_count = slice_mb_width;

        mb_x += slice_mb_width;

        if (mb_x == frame.mb_width)
            mb_x = 0, slice_mb_width = 1 << frame.log2_slice_mb_width;
    }

    AVPixelFormat pixfmt;
    switch ((this->depth << 16) | (static_cast<int>(frame.alpha_type) << 8) | (static_cast<int>(frame.chroma_fmt) << 0)) {
        case (10 << 16) | (static_cast<int>(ProresAlphaType::None) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
            pixfmt = AV_PIX_FMT_YUV422P10;
            break;
        case (10 << 16) | (static_cast<int>(ProresAlphaType::None) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
            pixfmt = AV_PIX_FMT_YUV444P10;
            break;
        case (10 << 16) | (static_cast<int>(ProresAlphaType::B8  ) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
        case (10 << 16) | (static_cast<int>(ProresAlphaType::B16 ) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
            pixfmt = AV_PIX_FMT_YUVA422P10;
            break;
        case (10 << 16) | (static_cast<int>(ProresAlphaType::B8  ) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
        case (10 << 16) | (static_cast<int>(ProresAlphaType::B16 ) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
            pixfmt = AV_PIX_FMT_YUVA444P10;
            break;
        case (12 << 16) | (static_cast<int>(ProresAlphaType::None) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
            pixfmt = AV_PIX_FMT_YUV422P12;
            break;
        case (12 << 16) | (static_cast<int>(ProresAlphaType::None) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
            pixfmt = AV_PIX_FMT_YUV444P12;
            break;
        case (12 << 16) | (static_cast<int>(ProresAlphaType::B8  ) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
        case (12 << 16) | (static_cast<int>(ProresAlphaType::B16 ) << 8) | (static_cast<int>(ProresChromaFormat::C422) << 0):
            pixfmt = AV_PIX_FMT_YUVA422P12;
            break;
        case (12 << 16) | (static_cast<int>(ProresAlphaType::B8  ) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
        case (12 << 16) | (static_cast<int>(ProresAlphaType::B16 ) << 8) | (static_cast<int>(ProresChromaFormat::C444) << 0):
            pixfmt = AV_PIX_FMT_YUVA444P12;
            break;
        default:
            return -1;
    }

    auto *pixdesc = av_pix_fmt_desc_get(pixfmt);

    std::array<cudaArray_t,         4> arrays   = {};
    std::array<cudaSurfaceObject_t, 4> surfobjs = {};
    SCOPEGUARD([&] { for (std::size_t i = 0; i < pixdesc->nb_components; ++i) cudaFreeArray           (arrays  [i]); });
    SCOPEGUARD([&] { for (std::size_t i = 0; i < pixdesc->nb_components; ++i) cudaDestroySurfaceObject(surfobjs[i]); });

    auto desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    for (std::size_t i = 0; i < pixdesc->nb_components; ++i) {
        bool chroma = std::clamp<std::size_t>(i, 1, 2) == i;
        auto w = util::align_up(frame.pic_width,  ProresMbSize) * (int(frame.interlace_mode != ProresInterlaceMode::Progressive) + 1),
             h = util::align_up(frame.pic_height, ProresMbSize) * (int(frame.interlace_mode != ProresInterlaceMode::Progressive) + 1);
        CUDA_CHECK(cudaMallocArray(&arrays[i], &desc,
                                   w >> (chroma ? pixdesc->log2_chroma_w : 0),
                                   h >> (chroma ? pixdesc->log2_chroma_h : 0),
                                   cudaArraySurfaceLoadStore));

        auto desc = cudaResourceDesc{
            .resType = cudaResourceTypeArray,
            .res = { .array = { .array = arrays[i], }, },
        };
        CUDA_CHECK(cudaCreateSurfaceObject(&surfobjs[i], &desc));
    }

    void *surfaces, *slice_offsets, *slice_ctxs, *slice_data;
    CUDA_CHECK(cudaMalloc(&surfaces,      pixdesc->nb_components   * sizeof(cudaSurfaceObject_t)));
    CUDA_CHECK(cudaMalloc(&slice_offsets, slice_offsets_cpu.size() * sizeof(std::uint32_t      )));
    CUDA_CHECK(cudaMalloc(&slice_ctxs,    slice_ctxs_cpu   .size() * sizeof(SliceContext       )));
    CUDA_CHECK(cudaMalloc(&slice_data,    frame.buf->size));
    SCOPEGUARD([&surfaces     ] { cudaFree(surfaces     ); });
    SCOPEGUARD([&slice_offsets] { cudaFree(slice_offsets); });
    SCOPEGUARD([&slice_ctxs   ] { cudaFree(slice_ctxs   ); });
    SCOPEGUARD([&slice_data   ] { cudaFree(slice_data   ); });

    auto p = KernelParams{
        .slice_data      = reinterpret_cast<CUdeviceptr>(slice_data) + bs.tell(),
        .bitstream_size  = static_cast<std::uint32_t>(frame.buf->size - bs.tell()),
        .width           = frame.pic_width,
        .height          = frame.pic_height,
        .mb_width        = frame.mb_width,
        .mb_height       = frame.mb_height,
        .slice_width     = frame.slice_width,
        .slice_height    = frame.mb_height,
        .log2_chroma_w   = static_cast<std::uint8_t>(frame.chroma_fmt == ProresChromaFormat::C422 ? 1 : 0),
        .depth           = static_cast<std::uint8_t>(this->depth),
        .alpha_info      = static_cast<std::uint8_t>(frame.alpha_type),
        .bottom_field    = static_cast<std::uint8_t>(frame.bottom_field)
    };

    std::copy_n(frame.luma_qmat  [0], sizeof(frame.luma_qmat  ) * sizeof(std::uint8_t), p.qmat_luma  [0]);
    std::copy_n(frame.chroma_qmat[0], sizeof(frame.chroma_qmat) * sizeof(std::uint8_t), p.qmat_chroma[0]);

    CUDA_CHECK(cudaMemcpy(surfaces,      surfobjs         .data(), pixdesc->nb_components   * sizeof(cudaSurfaceObject_t), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(slice_offsets, slice_offsets_cpu.data(), slice_offsets_cpu.size() * sizeof(std::uint32_t      ), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(slice_ctxs,    slice_ctxs_cpu   .data(), slice_ctxs_cpu   .size() * sizeof(SliceContext       ), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(slice_data,    frame.buf->data,  frame.buf->size,                                                cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpyToSymbol(params, &p, sizeof(p)));

    auto vld_grid_size   = dim3(util::ceil_rshift(frame.slice_width, 3), util::ceil_rshift(frame.mb_height, 3), 3),
         vld_block_size  = dim3(8, 8, 1);
    auto idct_grid_size  = dim3(util::ceil_rshift(frame.mb_width, 1), frame.mb_height, 3),
         idct_block_size = dim3(32, 2, 1);
    if (frame.interlace_mode == ProresInterlaceMode::Progressive) {
        kern_color_vld<false><<<vld_grid_size, vld_block_size>>>(
            static_cast<cudaSurfaceObject_t *>(surfaces), static_cast<std::uint32_t *>(slice_offsets), static_cast<SliceContext *>(slice_ctxs)
        );
        kern_idct<false><<<idct_grid_size, idct_block_size>>>(
            static_cast<cudaSurfaceObject_t *>(surfaces)
        );
        if (frame.alpha_type != ProresAlphaType::None)
            kern_alpha_vld<false><<<vld_grid_size, vld_block_size>>>(
                static_cast<cudaSurfaceObject_t *>(surfaces), static_cast<std::uint32_t *>(slice_offsets), static_cast<SliceContext *>(slice_ctxs)
            );
    } else {
        kern_color_vld<true><<<vld_grid_size, vld_block_size>>>(
            static_cast<cudaSurfaceObject_t *>(surfaces), static_cast<std::uint32_t *>(slice_offsets), static_cast<SliceContext *>(slice_ctxs)
        );
        kern_idct<true><<<idct_grid_size, idct_block_size>>>(
            static_cast<cudaSurfaceObject_t *>(surfaces)
        );
        if (frame.alpha_type != ProresAlphaType::None)
            kern_alpha_vld<true><<<vld_grid_size, vld_block_size>>>(
                static_cast<cudaSurfaceObject_t *>(surfaces), static_cast<std::uint32_t *>(slice_offsets), static_cast<SliceContext *>(slice_ctxs)
            );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < pixdesc->nb_components && dst->data[i]; ++i) {
        bool chroma = std::clamp<std::size_t>(i, 1, 2) == i;
        CUDA_CHECK(cudaMemcpy2DFromArray(dst->data[i], dst->linesize[i], arrays[i], 0, 0,
                                         frame.width * sizeof(std::uint16_t) >> (chroma ? pixdesc->log2_chroma_w : 0),
                                         frame.height                        >> (chroma ? pixdesc->log2_chroma_h : 0),
                                         cudaMemcpyDefault));
    }

    return 0;
}
