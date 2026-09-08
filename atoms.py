"""MP4 atom/box builders — constructs the moov atom and all sub-boxes.

Uses array.array for bulk packing of sample tables (stsz, stco, stss)
instead of per-element struct.pack + string concatenation.
"""

import sys
import array
import struct


def build_box(box_type, content):
    """Build MP4 box: [size:4][type:4][content]."""
    return struct.pack('>I4s', 8 + len(content), box_type) + content


def build_full_box(box_type, version, flags, content):
    """Build MP4 full box: [size:4][type:4][version+flags:4][content]."""
    return struct.pack('>I4sI', 12 + len(content), box_type,
                       (version << 24) | flags) + content


# ─── Track-level boxes ───────────────────────────────────────────────────────

def build_mvhd(timescale, duration):
    if duration > 0xFFFFFFFF:
        data = struct.pack(
            '>QQIQ', 0, 0, timescale, duration
        ) + struct.pack('>I', 0x00010000) + struct.pack(
            '>H', 0x0100) + b'\x00' * 10 + struct.pack(
            '>9I', 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000
        ) + b'\x00' * 24 + struct.pack('>I', 3)
        return build_full_box(b'mvhd', 1, 0, data)
    data = struct.pack(
        '>IIIII', 0, 0, timescale, duration, 0x00010000  # rate 1.0
    ) + struct.pack('>H', 0x0100) + b'\x00' * 10 + struct.pack(
        '>9I', 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000
    ) + b'\x00' * 24 + struct.pack('>I', 3)
    return build_full_box(b'mvhd', 0, 0, data)


def build_tkhd(track_id, duration, width, height, is_audio):
    if duration > 0xFFFFFFFF:
        data = b''.join([
            struct.pack('>QQI', 0, 0, track_id),
            b'\x00' * 4,                                    # reserved
            struct.pack('>Q', duration),
            b'\x00' * 8,                                    # reserved
            struct.pack('>HHH', 0, 0, 0x0100 if is_audio else 0),
            b'\x00' * 2,
            struct.pack('>9I', 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000),
            struct.pack('>II', width << 16, height << 16),
        ])
        return build_full_box(b'tkhd', 1, 3, data)
    data = b''.join([
        struct.pack('>III', 0, 0, track_id),
        b'\x00' * 4,                                    # reserved
        struct.pack('>I', duration),
        b'\x00' * 8,                                    # reserved
        struct.pack('>HHH', 0, 0, 0x0100 if is_audio else 0),
        b'\x00' * 2,
        struct.pack('>9I', 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000),
        struct.pack('>II', width << 16, height << 16),
    ])
    return build_full_box(b'tkhd', 0, 3, data)


def build_mdhd(timescale, duration):
    data = struct.pack('>IIII', 0, 0, timescale, duration)
    data += struct.pack('>HH', 0x55C4, 0)  # language='und'
    return build_full_box(b'mdhd', 0, 0, data)


def build_hdlr(handler_type, name):
    c = b'\x00' * 4 + handler_type + b'\x00' * 12 + name + b'\x00'
    return build_full_box(b'hdlr', 0, 0, c)


def build_vmhd():
    return build_full_box(b'vmhd', 0, 1, b'\x00' * 8)


def build_smhd():
    return build_full_box(b'smhd', 0, 0, b'\x00' * 4)


def build_dinf():
    url_box = build_full_box(b'url ', 0, 1, b'')
    dref = build_full_box(b'dref', 0, 0, struct.pack('>I', 1) + url_box)
    return build_box(b'dinf', dref)


# ─── Sample description boxes ───────────────────────────────────────────────

def build_stsd_video(sps, pps, width, height):
    """Build video stsd with avc1 + avcC."""
    avcc_data = b''.join([
        struct.pack('B', 1),
        sps[1:4] if len(sps) >= 4 else b'\x4d\x40\x28',
        struct.pack('BB', 0xFF, 0xE1),
        struct.pack('>H', len(sps)), sps,
        struct.pack('B', 1),
        struct.pack('>H', len(pps)), pps,
    ])
    avcc = build_box(b'avcC', avcc_data)

    avc1_data = b''.join([
        b'\x00' * 6,
        struct.pack('>H', 1),                        # data_reference_index
        b'\x00' * 16,
        struct.pack('>HH', width, height),
        struct.pack('>II', 0x00480000, 0x00480000),  # 72 dpi
        struct.pack('>IH', 0, 1),                    # data_size, frame_count
        b'\x0a' + b'AVC Coding' + b'\x00' * 21,     # compressor_name (32 bytes)
        struct.pack('>Hh', 0x0018, -1),              # depth=24, color_table=-1
        avcc,
    ])
    avc1 = build_box(b'avc1', avc1_data)
    return build_full_box(b'stsd', 0, 0, struct.pack('>I', 1) + avc1)


def build_stsd_audio(timescale, channels=2, audio_object_type=2):
    """Build audio stsd with mp4a + esds.

    The original implementation only needed AAC-LC at 48 kHz stereo.  Keep
    those defaults, but generate AudioSpecificConfig for other supported
    sample rates/channels so reference-free recovery can be configured for
    similar recordings.
    """
    sample_rates = {
        96000: 0, 88200: 1, 64000: 2, 48000: 3, 44100: 4,
        32000: 5, 24000: 6, 22050: 7, 16000: 8, 12000: 9,
        11025: 10, 8000: 11, 7350: 12,
    }
    if timescale not in sample_rates:
        raise ValueError(f'Unsupported AAC sample rate: {timescale}')
    if not 1 <= channels <= 7:
        raise ValueError(f'Unsupported AAC channel count: {channels}')
    # audioObjectType(5), samplingFrequencyIndex(4), channelConfig(4),
    # followed by GASpecificConfig frameLengthFlag(1), dependsOnCoreCoder(1),
    # extensionFlag(1), and padding.
    config_bits = ((audio_object_type & 0x1F) << 11) | \
                  (sample_rates[timescale] << 7) | (channels << 3)
    aac_config = struct.pack('>H', config_bits)

    def desc_tag(tag, content):
        return bytes([tag, 0x80, 0x80, 0x80, len(content)]) + content

    dec_spec = desc_tag(0x05, aac_config)
    dec_cfg = desc_tag(0x04, (
        b'\x40\x15\x00\x06\x00' +
        struct.pack('>II', 192000, 192000) +
        dec_spec
    ))
    es_desc = desc_tag(0x03, (
        struct.pack('>HB', 0, 0) +
        dec_cfg +
        b'\x06\x01\x02'
    ))
    esds = build_full_box(b'esds', 0, 0, es_desc)

    mp4a_data = b''.join([
        b'\x00' * 6,
        struct.pack('>H', 1),
        b'\x00' * 8,
        struct.pack('>HH', channels, 16),
        b'\x00' * 4,
        struct.pack('>I', timescale << 16),
        esds,
    ])
    mp4a = build_box(b'mp4a', mp4a_data)
    return build_full_box(b'stsd', 0, 0, struct.pack('>I', 1) + mp4a)


# ─── Sample table boxes (array.array for bulk packing) ───────────────────────

def build_stts(sample_count, sample_delta):
    return build_full_box(b'stts', 0, 0,
                          struct.pack('>III', 1, sample_count, sample_delta))


def build_ctts_from_types(slice_types, sample_delta):
    """Build ctts from per-sample slice types (1=B, 0=non-B).

    Version 0 (unsigned offsets) — all PTS shifted by +sample_delta:
      B-frame: offset=0, non-B before B: offset=2*delta, else: offset=delta
    """
    n = len(slice_types)
    if n == 0:
        return b''

    delta2 = 2 * sample_delta

    # Build run-length compressed entries directly into flat list
    entries = []  # [count1, offset1, count2, offset2, ...]
    prev_offset = -1
    count = 0
    for i in range(n):
        if slice_types[i] == 1:
            offset = 0
        elif i + 1 < n and slice_types[i + 1] == 1:
            offset = delta2
        else:
            offset = sample_delta

        if offset == prev_offset:
            count += 1
        else:
            if count > 0:
                entries.append(count)
                entries.append(prev_offset)
            prev_offset = offset
            count = 1
    if count > 0:
        entries.append(count)
        entries.append(prev_offset)

    num_entries = len(entries) // 2
    arr = array.array('I', entries)
    if sys.byteorder == 'little':
        arr.byteswap()
    return build_full_box(b'ctts', 0, 0,
                          struct.pack('>I', num_entries) + arr.tobytes())


def build_stsc(chunks):
    """Build stsc — compresses runs with same sample count."""
    entries = []
    prev_n = None
    for i, (_, sample_indices) in enumerate(chunks):
        n = len(sample_indices)
        if n != prev_n:
            entries.append((i + 1, n, 1))
            prev_n = n
    if not entries:
        entries = [(1, 1, 1)]
    data = struct.pack('>I', len(entries))
    for fc, spc, di in entries:
        data += struct.pack('>III', fc, spc, di)
    return build_full_box(b'stsc', 0, 0, data)


def build_stsz(samples):
    """Build stsz using array.array for O(1) bulk endian conversion."""
    n = len(samples)
    header = struct.pack('>II', 0, n)
    if n == 0:
        return build_full_box(b'stsz', 0, 0, header)
    arr = array.array('I', (s for _, s in samples))
    if sys.byteorder == 'little':
        arr.byteswap()
    return build_full_box(b'stsz', 0, 0, header + arr.tobytes())


def build_stco(chunks, use_co64=False):
    """Build stco/co64 using array.array for bulk packing."""
    n = len(chunks)
    if use_co64:
        arr = array.array('Q', (o for o, _ in chunks))
        if sys.byteorder == 'little':
            arr.byteswap()
        return build_full_box(b'co64', 0, 0,
                              struct.pack('>I', n) + arr.tobytes())
    else:
        arr = array.array('I', (o for o, _ in chunks))
        if sys.byteorder == 'little':
            arr.byteswap()
        return build_full_box(b'stco', 0, 0,
                              struct.pack('>I', n) + arr.tobytes())


def build_stss(sync_samples):
    """Build stss using array.array for bulk packing."""
    n = len(sync_samples)
    arr = array.array('I', sync_samples)
    if sys.byteorder == 'little':
        arr.byteswap()
    return build_full_box(b'stss', 0, 0,
                          struct.pack('>I', n) + arr.tobytes())


# ─── Composite boxes ─────────────────────────────────────────────────────────

def build_trak(track_id, tkhd_duration, media_timescale, media_duration,
               handler_type, handler_name, media_header_box,
               stsd, stts, stsc, stsz, stco, stss,
               width, height, is_audio, ctts=None):
    """Assemble a complete trak box."""
    tkhd = build_tkhd(track_id, tkhd_duration, width, height, is_audio)
    mdhd = build_mdhd(media_timescale, media_duration)
    hdlr = build_hdlr(handler_type, handler_name)
    dinf = build_dinf()

    stbl_parts = [stsd, stts, stsc, stsz, stco]
    if ctts is not None:
        stbl_parts.append(ctts)
    if stss is not None:
        stbl_parts.append(stss)
    stbl = build_box(b'stbl', b''.join(stbl_parts))

    minf = build_box(b'minf', media_header_box + dinf + stbl)
    mdia = build_box(b'mdia', mdhd + hdlr + minf)
    return build_box(b'trak', tkhd + mdia)


def build_moov(ref_info, scan_info):
    """Build the complete moov atom."""
    vs = scan_info['video_samples']
    as_ = scan_info['audio_samples']
    vc = scan_info['video_chunks']
    ac = scan_info['audio_chunks']
    ss = scan_info['sync_samples']

    mvhd_ts = ref_info['mvhd_timescale']
    v_ts = ref_info['video']['timescale']
    a_ts = ref_info['audio']['timescale']
    v_delta = ref_info['video']['sample_delta']
    a_delta = ref_info['audio']['sample_delta']

    v_media_dur = len(vs) * v_delta
    a_media_dur = len(as_) * a_delta
    v_tkhd_dur = v_media_dur * mvhd_ts // v_ts
    a_tkhd_dur = a_media_dur * mvhd_ts // a_ts if as_ else 0
    mvhd_dur = max(v_tkhd_dur, a_tkhd_dur)

    max_offset = 0
    if vc:
        max_offset = max(max_offset, max(o for o, _ in vc))
    if ac:
        max_offset = max(max_offset, max(o for o, _ in ac))
    use_co64 = max_offset > 0xFFFFFFFF

    mvhd = build_mvhd(mvhd_ts, mvhd_dur)

    # Video trak
    w = ref_info['video']['width']
    h = ref_info['video']['height']
    sps = ref_info['video'].get('sps', b'\x67\x4d\x40\x28')
    pps = ref_info['video'].get('pps', b'\x68\xce\x3c\x80')

    v_stsd = build_stsd_video(sps, pps, w, h)
    v_stts = build_stts(len(vs), v_delta)

    v_ctts = None
    if scan_info.get('has_b_frames'):
        v_ctts = build_ctts_from_types(scan_info['slice_types'], v_delta)

    v_stsc = build_stsc(vc)
    v_stsz = build_stsz(vs)
    v_stco = build_stco(vc, use_co64)
    v_stss = build_stss(ss if ss else [1])

    video_trak = build_trak(1, v_tkhd_dur, v_ts, v_media_dur,
                            b'vide', b'VideoHandler', build_vmhd(),
                            v_stsd, v_stts, v_stsc, v_stsz, v_stco, v_stss,
                            w, h, False, ctts=v_ctts)

    # Audio trak
    audio_trak = b''
    if as_:
        a_stsd = build_stsd_audio(a_ts,
                                   ref_info['audio'].get('channels', 2),
                                   ref_info['audio'].get('audio_object_type', 2))
        a_stts = build_stts(len(as_), a_delta)
        a_stsc = build_stsc(ac)
        a_stsz = build_stsz(as_)
        a_stco = build_stco(ac, use_co64)

        audio_trak = build_trak(2, a_tkhd_dur, a_ts, a_media_dur,
                                b'soun', b'SoundHandler', build_smhd(),
                                a_stsd, a_stts, a_stsc, a_stsz, a_stco, None,
                                0, 0, True)

    return build_box(b'moov', mvhd + video_trak + audio_trak)
