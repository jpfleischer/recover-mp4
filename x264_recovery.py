"""Reference-free recovery for the length-prefixed x264 recording variant.

The main scanner in this project targets Windows Snipping Tool files, whose
video access units begin with an AUD NAL.  Some screen recorders instead write
one length-prefixed H.264 slice followed by raw AAC frames and omit the AUD.
This module handles that layout and reuses the project's MP4 atom builder and
writer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
import mmap
import os
import re
import struct
import subprocess
from collections import Counter

try:
    from .atoms import build_moov
    from .scanner import _find_aac_boundaries
    from .writer import write_output
except ImportError:  # Also allow this recovery path to run as a script.
    from atoms import build_moov
    from scanner import _find_aac_boundaries
    from writer import write_output


@dataclass(frozen=True)
class X264Config:
    """Settings needed to rebuild the missing MP4 metadata.

    These defaults match the file investigated with this recovery.  The
    scanner can handle the same no-AUD layout at other dimensions and rates
    when the original recorder settings are supplied on the command line.
    """

    width: int = 1920
    height: int = 1080
    fps_num: int = 60
    fps_den: int = 1
    audio_rate: int = 48000
    audio_channels: int = 2
    audio_frame_samples: int = 1024
    audio_min_frame_bytes: int = 40
    audio_max_frame_bytes: int = 1200
    frame_num_bits: int = 4
    x264_cabac: int = 1
    x264_ref: int = 3
    x264_bframes: int = 3

    @property
    def video_timescale(self):
        return self.fps_num * 1000

    @property
    def video_delta(self):
        return self.fps_den * 1000


DEFAULT_CONFIG = X264Config()


def _read_ue(data: bytes, bit_pos: int) -> tuple[int, int] | None:
    """Read one unsigned Exp-Golomb value from data, returning (value, pos)."""
    total_bits = len(data) * 8
    zeros = 0
    while bit_pos < total_bits and not (data[bit_pos // 8] & (0x80 >> (bit_pos & 7))):
        zeros += 1
        bit_pos += 1
        if zeros > 16:
            return None
    if bit_pos >= total_bits:
        return None
    bit_pos += 1  # the one bit
    if bit_pos + zeros > total_bits:
        return None
    value = 0
    for _ in range(zeros):
        value = (value << 1) | bool(data[bit_pos // 8] & (0x80 >> (bit_pos & 7)))
        bit_pos += 1
    return ((1 << zeros) - 1 + value, bit_pos)


def _slice_header(buf, payload_pos: int, nal_size: int, frame_num_bits=4):
    """Return (first_mb, slice_type, pps_id, frame_num) for a slice NAL."""
    if nal_size < 5:
        return None
    if not 1 <= frame_num_bits <= 16:
        return None
    # The first fields are before any PPS-dependent fields. The configured
    # frame_num width is small enough that 12 bytes covers the header.
    data = bytes(buf[payload_pos:payload_pos + min(nal_size - 1, 12)])
    p = 0
    values = []
    for _ in range(3):
        item = _read_ue(data, p)
        if item is None:
            return None
        value, p = item
        values.append(value)
    if p + frame_num_bits > len(data) * 8:
        return None
    frame_num = 0
    for _ in range(frame_num_bits):
        frame_num = (frame_num << 1) | bool(data[p // 8] & (0x80 >> (p & 7)))
        p += 1
    return values[0], values[1], values[2], frame_num


def _mdat_info(path: str):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        pos = 0
        while pos + 8 <= size:
            f.seek(pos)
            raw = f.read(8)
            box_size, box_type = struct.unpack('>I4s', raw)
            header = 8
            if box_size == 1:
                extended = f.read(8)
                if len(extended) != 8:
                    break
                box_size = struct.unpack('>Q', extended)[0]
                header = 16
            elif box_size == 0:
                box_size = size - pos
            if box_type == b'mdat':
                return pos, box_size, pos + header, min(pos + box_size, size)
            if box_size < header:
                break
            pos += box_size
    raise ValueError('No mdat box found')


def _candidate(view, q: int, end: int, frame_num_bits=4):
    """Parse a plausible length-prefixed NAL beginning at absolute q."""
    # Real NAL lengths are commonly 0x0001xxxx here; only the high byte is
    # reliably zero, not all four length-prefix bytes.
    if q + 5 > end or view[q] != 0:
        return None
    length = struct.unpack_from('>I', view, q)[0]
    if length < 5 or length > 8_000_000 or q + 4 + length > end:
        return None
    header = view[q + 4]
    if header & 0x80:
        return None
    nal_type = header & 0x1F
    if nal_type not in (1, 5, 6):
        return None
    if nal_type == 6:
        return length, nal_type, None
    # A false length prefix often points deep into a true IDR.  Keep the cap
    # generous for real 1080p frames but reject implausible P/B lengths.
    if nal_type == 1 and length > 500_000:
        return None
    if nal_type == 5 and length > 3_000_000:
        return None
    hdr = _slice_header(view, q + 5, length, frame_num_bits)
    if hdr is None or hdr[0] != 0 or hdr[2] != 0:
        return None
    # x264's real IDR slices in this stream use the all-intra slice_type 7.
    # A tiny type-5 candidate is a common false positive in AAC bytes.
    if nal_type == 5 and (hdr[1] != 7 or length < 10_000):
        return None
    # The encoder writes the modulo-5 slice_type values 5/6/7 (P/B/I).
    # Values 0/1/2 are valid H.264 syntax too, but in this byte stream they
    # are overwhelmingly false positives inside AAC or a larger slice.
    if nal_type == 1 and hdr[1] not in (5, 6, 7):
        return None
    return length, nal_type, hdr


def _has_inner_slice(view, q: int, length: int, end: int, frame_num_bits=4) -> bool:
    """Reject a suspicious giant IDR candidate containing another NAL start."""
    search_end = min(q + 5 + length, q + 5 + 1024, end - 5)
    p = q + 5
    while p < search_end:
        p = view.find(b'\x00', p, search_end)
        if p < 0:
            return False
        inner = _candidate(view, p, end, frame_num_bits)
        if inner is not None and inner[1] in (1, 5):
            return True
        p += 1
    return False


def _has_plausible_successor(view, q, length, nal_type, st, frame_num, end,
                             frame_num_bits=4):
    """Check that a candidate is followed by the next video NAL promptly.

    In this recording an AAC frame is at most about 1 KB, so a real video NAL
    is followed by another slice within a small bounded gap.  This lookahead
    is important because random AAC bytes can occasionally resemble a valid
    H.264 slice header and advertise a very large false length.
    """
    current_end = q + 4 + length
    search_end = min(end - 5, current_end + 1800)
    p = current_end
    current_is_idr = nal_type == 5 or st == 7
    while p < search_end:
        p = view.find(b'\x00', p, search_end)
        if p < 0:
            return False
        item = _candidate(view, p, end, frame_num_bits)
        if item is None:
            p += 1
            continue
        next_length, next_type, next_hdr = item
        if next_type == 6:
            p += 1
            continue
        if next_type == 5 and _has_inner_slice(
                view, p, next_length, end, frame_num_bits):
            p += 1
            continue
        next_frame = next_hdr[3]
        if next_type == 5 or current_is_idr:
            return next_type in (1, 5)
        delta = (next_frame - frame_num) % (1 << frame_num_bits)
        frame_modulus = 1 << frame_num_bits
        if delta in (0, 1, 2, frame_modulus - 1):
            return True
        p += 1
    return False


def _scan(path: str, config: X264Config = DEFAULT_CONFIG,
          verbose: bool = True):
    """Find video samples and interleaved audio regions in the raw mdat."""
    mdat_offset, mdat_size, data_start, data_end = _mdat_info(path)
    video_samples = []
    video_chunks = []
    slice_types = []
    sync_samples = []
    audio_regions = []
    rejected_idr = []

    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        view = mm
        first = _candidate(view, data_start, data_end, config.frame_num_bits)
        pending_start = None
        cursor = data_start
        prev_frame = None
        prev_was_idr = False
        last_video_end = None
        if first is not None and first[1] == 6:
            # The x264 encoder-info SEI belongs to the first access unit.
            pending_start = data_start
            cursor = data_start + 4 + first[0]

        while cursor < data_end:
            found = None
            p = cursor
            while p < data_end:
                p = view.find(b'\x00', p, data_end - 4)
                if p < 0:
                    break
                item = _candidate(view, p, data_end, config.frame_num_bits)
                if item is None:
                    p += 1
                    continue
                length, nal_type, hdr = item
                if nal_type == 6:
                    p += 1
                    continue
                if nal_type == 5 and _has_inner_slice(
                        view, p, length, data_end, config.frame_num_bits):
                    rejected_idr.append((p, length, hdr))
                    p += 1
                    continue

                first_mb, st, pps, frame_num = hdr
                is_idr = nal_type == 5 or st == 7
                near_end = p + 4 + length >= data_end - 12_000
                if not near_end and not _has_plausible_successor(
                        view, p, length, nal_type, st, frame_num, data_end,
                        config.frame_num_bits):
                    p += 1
                    continue
                accept = False
                if prev_frame is None:
                    accept = is_idr
                elif nal_type == 5:
                    # IDR pictures reset frame_num; continuity checks do not
                    # apply across an IDR boundary.
                    accept = True
                elif prev_was_idr:
                    # The first post-IDR picture can reuse frame_num zero.
                    accept = nal_type == 1
                else:
                    delta = (frame_num - prev_frame) % (1 << config.frame_num_bits)
                    accept = delta in (0, 1, 2, (1 << config.frame_num_bits) - 1)
                if accept:
                    found = (p, length, nal_type, st, frame_num)
                    break
                p += 1

            if found is None:
                break
            q, length, nal_type, st, frame_num = found
            sample_start = pending_start if pending_start is not None else q
            sample_end = q + 4 + length
            video_samples.append((sample_start, sample_end - sample_start))
            video_chunks.append((sample_start, [len(video_samples) - 1]))
            slice_types.append(1 if st in (1, 6) else 0)
            if is_idr := (nal_type == 5 or st == 7):
                sync_samples.append(len(video_samples))
            if last_video_end is not None and q > last_video_end:
                audio_regions.append((last_video_end, q))
            last_video_end = sample_end
            pending_start = None
            prev_frame = frame_num
            prev_was_idr = is_idr
            cursor = sample_end

        if last_video_end is not None and last_video_end < data_end:
            audio_regions.append((last_video_end, data_end))

    # Probe output is intentionally compact: it is useful for checking that
    # the heuristic did not swallow a large part of the file as a false NAL.
    if verbose:
        gaps = [b - a for a, b in audio_regions if b > a]
        print(f'  Video samples: {len(video_samples):,}')
        print(f'  Audio regions: {len(audio_regions):,}')
        print(f'  Keyframes: {len(sync_samples):,}')
        print(f'  B-frames: {sum(slice_types):,}')
        print(f'  Largest audio regions: {sorted(gaps, reverse=True)[:8]}')
        if rejected_idr:
            print(f'  Rejected suspicious IDR candidates: {len(rejected_idr)}')

    return {
        'video_samples': video_samples,
        'video_chunks': video_chunks,
        'slice_types': slice_types,
        'sync_samples': sync_samples,
        'audio_regions': audio_regions,
        'mdat_offset': mdat_offset,
        'mdat_size': mdat_size,
        '_rejected_idr': rejected_idr,
    }


def _make_sps_pps(config: X264Config = DEFAULT_CONFIG):
    """Generate a matching High-profile SPS/PPS using the installed x264."""
    ffmpeg = 'ffmpeg'
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i',
        f'color=s={config.width}x{config.height}:r={config.fps_num}/{config.fps_den}',
        '-frames:v', '1', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-x264-params',
        f'cabac={config.x264_cabac}:ref={config.x264_ref}:bframes={config.x264_bframes}',
        '-f', 'h264', 'pipe:1',
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    data = result.stdout
    nals = []
    starts = list(re.finditer(b'\x00\x00(?:\x00)?\x01', data))
    for i, marker in enumerate(starts):
        stop = starts[i + 1].start() if i + 1 < len(starts) else len(data)
        nal = data[marker.end():stop]
        if nal:
            nals.append(nal)
    sps = next((n for n in nals if n[0] & 0x1F == 7), None)
    pps = next((n for n in nals if n[0] & 0x1F == 8), None)
    if not sps or not pps:
        raise RuntimeError('Could not generate SPS/PPS with ffmpeg/libx264')
    return sps, pps


COMMON_RESOLUTIONS = (
    (7680, 4320), (5120, 2880), (4096, 2160), (3840, 2160),
    (2560, 1440), (1920, 1200), (1920, 1080), (1680, 1050),
    (1600, 900), (1440, 900), (1366, 768), (1280, 800),
    (1280, 720), (1024, 768), (960, 540), (854, 480),
    (800, 600), (640, 480), (640, 360), (426, 240),
)


def _parse_x264_options(data: bytes):
    """Read numeric x264 options from its encoder-info SEI, when present."""
    match = re.search(rb'x264 - core [^\x00]*? - options: ([^\x00]+)', data)
    if match is None:
        return {}
    options = {}
    for name, value in re.findall(
            rb'(?:^| )(cabac|ref|bframes|keyint_min)=([0-9]+)',
            match.group(1)):
        options[name.decode()] = int(value)
    return options


def _read_x264_options(path: str):
    """Read the small prefix where x264 normally writes its SEI string."""
    _, _, data_start, data_end = _mdat_info(path)
    with open(path, 'rb') as f:
        f.seek(data_start)
        return _parse_x264_options(f.read(min(64 << 10, data_end - data_start)))


def _video_probe_stream(source: str, scan, max_samples=8):
    """Convert a few length-prefixed video samples to an Annex-B probe stream."""
    out = bytearray()
    with open(source, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for offset, size in scan['video_samples'][:max_samples]:
            pos = offset
            stop = offset + size
            while pos + 4 <= stop:
                nal_size = struct.unpack_from('>I', mm, pos)[0]
                pos += 4
                if nal_size < 1 or pos + nal_size > stop:
                    break
                out += b'\x00\x00\x00\x01' + mm[pos:pos + nal_size]
                pos += nal_size
    return bytes(out)


def _decodes_probe(stream: bytes, config: X264Config):
    """Return whether ffmpeg decodes the probe without H.264 errors."""
    sps, pps = _make_sps_pps(config)
    annex_b = b'\x00\x00\x00\x01' + sps + b'\x00\x00\x00\x01' + pps + stream
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-v', 'error', '-f', 'h264', '-i', 'pipe:0',
         '-frames:v', '8', '-f', 'null', '-'],
        input=annex_b, capture_output=True, timeout=60)
    return result.returncode == 0 and not result.stderr.strip()


def _probe_dimensions(source: str, scan, config: X264Config):
    """Choose a common resolution whose generated SPS decodes the real slices."""
    stream = _video_probe_stream(source, scan)
    if not stream:
        return None
    resolutions = [(config.width, config.height)] + [
        pair for pair in COMMON_RESOLUTIONS
        if pair != (config.width, config.height)
    ]
    for width, height in resolutions:
        candidate = replace(config, width=width, height=height)
        try:
            if _decodes_probe(stream, candidate):
                return width, height
        except (OSError, RuntimeError, subprocess.SubprocessError):
            continue
    return None


def _infer_audio_rate(video_count: int, audio_count: int,
                      config: X264Config):
    """Infer a standard AAC rate from interleave counts and the video rate."""
    if not video_count or not audio_count:
        return None
    video_fps = config.fps_num / config.fps_den
    estimate = audio_count * config.audio_frame_samples * video_fps / video_count
    rates = (7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000,
             44100, 48000, 64000, 88200, 96000)
    rate = min(rates, key=lambda value: abs(value - estimate))
    return rate if abs(rate - estimate) / rate <= 0.03 else None


def _auto_config(source: str):
    """Best-effort configuration discovery for this raw x264 layout.

    The x264 SEI can reveal encoder options, and a short decode probe can
    distinguish common dimensions.  FPS and AAC rate are inferred from the
    interleave only when the result lands close to a standard value; the raw
    stream does not contain timestamps or an AAC sample-rate field.
    """
    options = _read_x264_options(source)
    config = replace(
        DEFAULT_CONFIG,
        x264_cabac=options.get('cabac', DEFAULT_CONFIG.x264_cabac),
        x264_ref=options.get('ref', DEFAULT_CONFIG.x264_ref),
        x264_bframes=options.get('bframes', DEFAULT_CONFIG.x264_bframes),
    )
    keyint_min = options.get('keyint_min')
    fps_source = 'default (not present in the raw stream)'
    if keyint_min is not None and 1 < keyint_min <= 240:
        # x264 commonly emits fps-1 for keyint_min in this recorder family.
        config = replace(config, fps_num=keyint_min + 1, fps_den=1)
        fps_source = f'keyint_min={keyint_min} heuristic'

    scan = _scan(source, config, verbose=False)
    dimensions = _probe_dimensions(source, scan, config)
    dimension_source = 'default (decode probe found no common match)'
    if dimensions is not None:
        config = replace(config, width=dimensions[0], height=dimensions[1])
        dimension_source = 'decode probe'

    with open(source, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        audio_ref = _audio_ref(mm, scan['audio_regions'], config)
        audio_samples, _ = _split_audio(mm, scan['audio_regions'], audio_ref, config)
    config = replace(config, audio_channels=audio_ref['channels'])
    audio_rate = _infer_audio_rate(
        len(scan['video_samples']), len(audio_samples), config)
    audio_rate_source = 'default (raw AAC has no sample-rate field)'
    if audio_rate is not None:
        config = replace(config, audio_rate=audio_rate)
        audio_rate_source = 'interleave-count heuristic'

    x264_label = 'Auto-detected' if options else 'Using default'
    print(f'  {x264_label} x264 options: '
          f'cabac={config.x264_cabac}, ref={config.x264_ref}, '
          f'bframes={config.x264_bframes}')
    print(f'  Dimensions: {config.width}x{config.height} ({dimension_source})')
    print(f'  Video rate: {config.fps_num}/{config.fps_den} fps ({fps_source})')
    print(f'  Auto-detected audio channels: {config.audio_channels}')
    print(f'  Audio rate: {config.audio_rate} Hz ({audio_rate_source})')
    print('  Assumed AAC frame length: '
          f'{config.audio_frame_samples} samples (not encoded in raw AAC)')
    return config


def _aac_header_info(data: bytes):
    if len(data) < 3 or (data[0] >> 5) not in (0, 1) or data[1] & 0x80:
        return None
    b0, b1, b2 = data[:3]
    channels = 2 if (b0 >> 5) == 1 else 1
    ws = (b1 >> 5) & 3
    if ws == 2:
        msf = b1 & 0x0F
        if msf > 14:
            return None
        long_window = False
    else:
        msf = ((b1 & 0x0F) << 2) | (b2 >> 6)
        if msf > 49 or b2 & 0x20:
            return None
        long_window = True
    ms = None if b0 == 0x20 or not long_window else (b2 >> 3) & 3
    if ms == 3:
        return None
    return msf, (msf if not long_window else None), ms, ws, channels


def _audio_ref(mm, regions, config: X264Config = DEFAULT_CONFIG):
    sizes = []
    long_counts = Counter()
    short_counts = Counter()
    ms_counts = Counter()
    channel_counts = Counter()
    for a, b in regions:
        n = b - a
        if config.audio_min_frame_bytes <= n <= config.audio_max_frame_bytes:
            info = _aac_header_info(bytes(mm[a:a + 3]))
            if info is None:
                continue
            sizes.append(n)
            msf, short_msf, ms, ws, channels = info
            channel_counts[channels] += 1
            if ws == 2:
                short_counts[short_msf] += 1
            else:
                long_counts[msf] += 1
                if ms is not None:
                    ms_counts[ms] += 1
    if not sizes:
        raise RuntimeError('Could not identify any raw AAC frames')
    mean = sum(sizes) / len(sizes)
    stdev = (sum((s - mean) ** 2 for s in sizes) / len(sizes)) ** 0.5
    return {
        'frame_size_min': min(sizes),
        'frame_size_max': max(sizes),
        'frame_size_mean': mean,
        'frame_size_stdev': stdev,
        'dominant_msf_long': long_counts.most_common(1)[0][0] if long_counts else None,
        'dominant_msf_short': short_counts.most_common(1)[0][0] if short_counts else None,
        'dominant_ms': ms_counts.most_common(1)[0][0] if ms_counts else None,
        'channels': channel_counts.most_common(1)[0][0] if channel_counts else 2,
    }


def _split_audio(mm, regions, audio_ref,
                 config: X264Config = DEFAULT_CONFIG):
    """Turn inter-video gaps into raw AAC samples."""
    samples = []
    chunks = []
    mean = audio_ref['frame_size_mean']
    for a, b in regions:
        size = b - a
        if size < config.audio_min_frame_bytes:
            continue
        if size <= config.audio_max_frame_bytes:
            samples.append((a, size))
            chunks.append((a, [len(samples) - 1]))
            continue
        count = max(2, round(size / mean))
        raw = bytes(mm[a:b])
        bounds = _find_aac_boundaries(raw, count, audio_ref)
        if not bounds or len(bounds) != count + 1:
            print(f'  Warning: equal-splitting audio region {size:,} bytes')
            bounds = [round(i * size / count) for i in range(count + 1)]
        indices = []
        for i in range(count):
            start, stop = a + bounds[i], a + bounds[i + 1]
            if stop <= start:
                continue
            samples.append((start, stop - start))
            indices.append(len(samples) - 1)
        if indices:
            chunks.append((a, indices))
    return samples, chunks


def _repair_audio_in_place(output: str, config: X264Config = DEFAULT_CONFIG):
    """Re-encode the complete audio track without per-segment duration drift."""
    ffmpeg = 'ffmpeg'
    temp = output + '.audio.tmp.mp4'
    cmd = [
        ffmpeg, '-y', '-hide_banner', '-v', 'error',
        '-err_detect', 'ignore_err', '-i', output,
        '-map', '0:v:0', '-map', '0:a:0',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        '-ar', str(config.audio_rate), '-ac', str(config.audio_channels),
        '-shortest', '-movflags', '+faststart', temp,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or not os.path.exists(temp):
        if os.path.exists(temp):
            os.remove(temp)
        raise RuntimeError(f'Audio re-encode failed: {result.stderr[-500:]}')
    os.replace(temp, output)


def _adjust_chunk_offsets(source: str, scan):
    """Account for an 8-to-16-byte mdat header change in the output."""
    with open(source, 'rb') as f:
        f.seek(scan['mdat_offset'])
        original_size_field = struct.unpack('>I', f.read(4))[0]
    original_header = 16 if original_size_field == 1 else 8
    output_header = 16 if scan['mdat_size'] > 0xFFFFFFFF else 8
    delta = output_header - original_header
    if delta:
        scan['video_chunks'] = [
            (offset + delta, samples) for offset, samples in scan['video_chunks']
        ]
        scan['audio_chunks'] = [
            (offset + delta, samples) for offset, samples in scan['audio_chunks']
        ]
        print(f'  mdat header: {original_header}B->{output_header}B; '
              f'adjusted chunk offsets by {delta:+d}')


def recover(source: str, output: str,
            config: X264Config = DEFAULT_CONFIG, scan_only: bool = False):
    scan = _scan(source, config)
    if scan_only:
        return
    sps, pps = _make_sps_pps(config)
    with open(source, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        audio_ref = _audio_ref(mm, scan['audio_regions'], config)
        audio_samples, audio_chunks = _split_audio(
            mm, scan['audio_regions'], audio_ref, config)
    scan['audio_samples'] = audio_samples
    scan['audio_chunks'] = audio_chunks
    scan['has_b_frames'] = any(scan['slice_types'])
    _adjust_chunk_offsets(source, scan)
    ref = {
        'mvhd_timescale': config.video_timescale,
        'video': {
            'width': config.width, 'height': config.height,
            'timescale': config.video_timescale,
            'sample_delta': config.video_delta,
            'sps': sps, 'pps': pps,
        },
        'audio': {
            'timescale': config.audio_rate,
            'sample_delta': config.audio_frame_samples,
            'samples_per_chunk': 1,
            'channels': config.audio_channels,
            'audio_object_type': 2,
        },
    }
    moov = build_moov(ref, scan)
    print(f'  AAC samples: {len(audio_samples):,}')
    print(f'  moov size: {len(moov):,} bytes')
    write_output(source, output, scan, moov)
    print('  Re-encoding the complete AAC track to keep timestamps continuous...')
    _repair_audio_in_place(output, config)
    print('  Audio repaired.')
    print(f'\nRecovery complete: {output}')
    duration_s = (len(scan['video_samples']) * config.fps_den /
                  config.fps_num)
    print(f'  Duration: {duration_s / 60:.1f} minutes')


def _positive_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be an integer') from exc
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def _fps(text):
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError('must be a number or fraction') from exc
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description='Recover an x264 MP4 with a missing moov atom')
    parser.add_argument('source')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--auto', action='store_true',
                        help='probe x264 metadata and common resolutions')
    parser.add_argument('--width', type=_positive_int)
    parser.add_argument('--height', type=_positive_int)
    parser.add_argument('--fps', type=_fps,
        help='video rate, e.g. 60 or 30000/1001')
    parser.add_argument('--audio-rate', type=_positive_int)
    parser.add_argument('--audio-channels', type=_positive_int)
    parser.add_argument('--audio-frame-samples', type=_positive_int)
    parser.add_argument('--audio-max-frame-bytes', type=_positive_int,
                        help='largest single raw AAC frame to treat as one sample')
    args = parser.parse_args()
    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output or os.path.splitext(source)[0] + '_recovered.mp4')
    if not os.path.exists(source):
        parser.error(f'file not found: {source}')
    if os.path.exists(output) and not args.scan_only:
        parser.error(f'output already exists; choose another path: {output}')
    config = _auto_config(source) if args.auto else DEFAULT_CONFIG
    overrides = {
        name: value for name, value in {
            'width': args.width,
            'height': args.height,
            'fps_num': args.fps.numerator if args.fps is not None else None,
            'fps_den': args.fps.denominator if args.fps is not None else None,
            'audio_rate': args.audio_rate,
            'audio_channels': args.audio_channels,
            'audio_frame_samples': args.audio_frame_samples,
            'audio_max_frame_bytes': args.audio_max_frame_bytes,
        }.items() if value is not None
    }
    config = replace(config, **overrides)
    if not 1 <= config.audio_channels <= 7:
        parser.error('--audio-channels must be between 1 and 7')
    recover(source, output, config, args.scan_only)


if __name__ == '__main__':
    main()
