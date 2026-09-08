"""mdat scanner — finds video and audio samples in corrupted MP4 data.

Scans interleaved video/audio data in a corrupted Snipping Tool MP4:
  [ftyp][uuid][mdat: [8-byte preamble][V chunk][A chunk][V chunk][A chunk]...]

Video chunks contain length-prefixed H.264 NAL units.
Audio chunks contain raw AAC-LC frames (no length prefix).

Performance: O(file_size) single-pass with buffered I/O.
"""

import struct
from bisect import bisect_left, bisect_right

try:
    from .constants import (
        AUD_PATTERN, VALID_NAL_MASK, MAX_NAL_SIZE, parse_slice_type,
    )
except ImportError:  # Also allow the recovery helpers to run as scripts.
    from constants import (
        AUD_PATTERN, VALID_NAL_MASK, MAX_NAL_SIZE, parse_slice_type,
    )

_BUF_SIZE = 4 * 1024 * 1024  # 4 MB read buffer
_UNPACK_I = struct.Struct('>I').unpack_from  # pre-compiled for hot loop


def scan_mdat(corrupted_path, ref_info):
    """Scan corrupted mdat for interleaved video and audio data.

    Returns dict with video_samples, audio_samples, video_chunks,
    audio_chunks, sync_samples, slice_types, has_b_frames,
    mdat_offset, mdat_size.
    """
    audio_ref = ref_info['audio']
    spc = audio_ref['samples_per_chunk']
    _needs_bootstrap = ('dominant_msf_long' not in audio_ref and
                        'frame_size_mean' not in audio_ref)

    with open(corrupted_path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        # Locate mdat atom
        mdat_offset = None
        mdat_size = 0
        while f.tell() < file_size:
            pos = f.tell()
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            size, btype = struct.unpack('>I4s', hdr)
            if size == 1:
                size = struct.unpack('>Q', f.read(8))[0]
            elif size == 0:
                size = file_size - pos
            if btype == b'mdat':
                mdat_offset = pos
                mdat_size = size
                break
            if size < 8:
                break
            f.seek(pos + size)

        if mdat_offset is None:
            raise ValueError("No mdat atom found")

        mdat_end = mdat_offset + mdat_size
        data_start = mdat_offset + 16  # skip mdat header + 8-byte preamble
        print(f"  mdat: offset={mdat_offset}, size={mdat_size / 1024 / 1024:.1f} MB")

        # Result arrays
        video_samples = []
        audio_samples = []
        video_chunks = []
        audio_chunks = []
        sync_samples = []    # 1-based sample indices of IDR frames
        slice_types = []     # per-sample: 1=B, 0=non-B
        has_b_frames = False

        # Read buffer — avoids per-NAL seek+read (~500K syscalls → ~500)
        buf = b''
        buf_start = 0
        buf_end = 0  # buf_start + len(buf), cached to avoid len() in hot loop

        # Access unit state
        au_start = 0
        au_size = 0
        au_is_sync = False
        au_has_slice = False
        au_slice_hdr = None
        chunk_sample_indices = []

        def finalize_au():
            """Finalize current access unit → append to video_samples."""
            nonlocal has_b_frames
            if au_size > 0 and au_has_slice:
                is_b = 0
                if au_slice_hdr is not None:
                    st = parse_slice_type(au_slice_hdr)
                    if st in (1, 6):
                        is_b = 1
                        has_b_frames = True
                video_samples.append((au_start, au_size))
                slice_types.append(is_b)
                chunk_sample_indices.append(len(video_samples) - 1)
                if au_is_sync:
                    sync_samples.append(len(video_samples))

        pos = data_start
        print("  Scanning mdat for video/audio samples...")

        while pos < mdat_end - 5:
            # ── Parse one video chunk ──
            chunk_offset = pos
            chunk_sample_indices = []
            au_start = pos
            au_size = 0
            au_is_sync = False
            au_has_slice = False
            au_slice_hdr = None

            while pos < mdat_end - 4:
                # Buffered read: refill when position falls outside buffer
                if pos < buf_start or pos + 9 > buf_end:
                    f.seek(pos)
                    buf = f.read(_BUF_SIZE)
                    buf_start = pos
                    buf_end = pos + len(buf)
                local = pos - buf_start
                if pos + 5 > buf_end:
                    break

                nal_len = _UNPACK_I(buf, local)[0]
                nal_byte = buf[local + 4]
                nal_type = nal_byte & 0x1F

                if (nal_len < 1 or nal_len > MAX_NAL_SIZE or
                        not (VALID_NAL_MASK >> nal_type) & 1 or
                        nal_byte & 0x80 or
                        pos + 4 + nal_len > mdat_end):
                    break

                nal_total = 4 + nal_len

                if nal_type == 9:  # AUD → new access unit
                    finalize_au()
                    au_start = pos
                    au_size = nal_total
                    au_is_sync = False
                    au_has_slice = False
                    au_slice_hdr = None
                else:
                    au_size += nal_total
                    if nal_type == 5:  # IDR slice
                        au_is_sync = True
                        au_has_slice = True
                        if au_slice_hdr is None and pos + 9 <= buf_end:
                            au_slice_hdr = buf[local + 5:local + 9]
                    elif nal_type <= 4:  # Coded slice (types 1-4)
                        au_has_slice = True
                        if au_slice_hdr is None and pos + 9 <= buf_end:
                            au_slice_hdr = buf[local + 5:local + 9]

                pos += nal_total

            finalize_au()

            if chunk_sample_indices:
                video_chunks.append((chunk_offset, chunk_sample_indices))

            # ── Find next video chunk (skip audio region) ──
            audio_start = pos
            next_video = _find_next_video(f, pos, mdat_end)

            if next_video is None:
                if pos < mdat_end:
                    _create_audio_chunk(f, audio_start, mdat_end, spc,
                                        audio_samples, audio_chunks, audio_ref)
                break

            if next_video > audio_start:
                _create_audio_chunk(f, audio_start, next_video, spc,
                                    audio_samples, audio_chunks, audio_ref)

            # Bootstrap audio stats from first chunks (reference-free mode)
            if _needs_bootstrap and len(audio_chunks) >= 10:
                _bootstrap_audio_ref(f, audio_samples, audio_chunks,
                                     audio_ref, spc)
                # Re-process first chunks with proper stats
                chunk_regions = []
                for _, si in audio_chunks:
                    s0 = audio_samples[si[0]]
                    sN = audio_samples[si[-1]]
                    chunk_regions.append((s0[0], sN[0] + sN[1]))
                audio_samples.clear()
                audio_chunks.clear()
                for cstart, cend in chunk_regions:
                    _create_audio_chunk(f, cstart, cend, spc,
                                        audio_samples, audio_chunks, audio_ref)
                _needs_bootstrap = False

            pos = next_video

            if len(video_chunks) % 500 == 0:
                pct = (pos - data_start) / (mdat_end - data_start) * 100
                print(f"    {len(video_samples):,} video, "
                      f"{len(audio_samples):,} audio samples ({pct:.1f}%)")

        print(f"  Found {len(video_samples):,} video samples "
              f"in {len(video_chunks):,} chunks")
        print(f"  Found {len(audio_samples):,} audio samples "
              f"in {len(audio_chunks):,} chunks")
        print(f"  Keyframes: {len(sync_samples):,}")
        b_count = sum(slice_types)
        print(f"  B-frames detected: {b_count} / {len(slice_types)}")

        return {
            'video_samples': video_samples,
            'audio_samples': audio_samples,
            'video_chunks': video_chunks,
            'audio_chunks': audio_chunks,
            'sync_samples': sync_samples,
            'slice_types': slice_types,
            'has_b_frames': has_b_frames,
            'mdat_offset': mdat_offset,
            'mdat_size': mdat_size,
        }


# ─── Video chunk detection ──────────────────────────────────────────────────

def _validate_video(buf, off, avail, file_pos, file_end):
    """Validate video chunk start at buf[off] using NAL header checks.

    Requires AUD (length=2, type=9) followed by at least 1 valid NAL header.
    Optionally validates a 2nd NAL header if the 1st fits in the buffer.
    """
    check_len = min(64, avail, file_end - file_pos)
    if check_len < 11:
        return False

    # First NAL must be AUD: length=2, type=9
    if _UNPACK_I(buf, off)[0] != 2:
        return False
    b4 = buf[off + 4]
    if b4 & 0x1F != 9 or b4 & 0x80:
        return False

    # Second NAL header at offset 6
    nl = _UNPACK_I(buf, off + 6)[0]
    nt = buf[off + 10] & 0x1F
    if (nl < 1 or nl > MAX_NAL_SIZE or
            not (VALID_NAL_MASK >> nt) & 1 or
            buf[off + 10] & 0x80):
        return False
    if file_pos + 10 + nl > file_end:
        return False

    # Optional 3rd NAL header if 2nd NAL fits in buffer
    p = 10 + nl
    if p + 5 <= check_len:
        nl2 = _UNPACK_I(buf, off + p)[0]
        nt2 = buf[off + p + 4] & 0x1F
        if (nl2 < 1 or nl2 > MAX_NAL_SIZE or
                not (VALID_NAL_MASK >> nt2) & 1 or
                buf[off + p + 4] & 0x80):
            return False
        if file_pos + p + 4 + nl2 > file_end:
            return False

    return True


def _find_next_video(f, start, end):
    """Scan forward from start to find the next video chunk start.

    Uses C-level bytes.find for AUD pattern search, then validates
    candidates directly from the search block (no extra I/O when possible).
    """
    pos = start
    while pos < end:
        f.seek(pos)
        block = f.read(min(_BUF_SIZE, end - pos))
        if not block:
            break
        block_len = len(block)
        if block_len < len(AUD_PATTERN):
            break

        idx = 0
        while True:
            idx = block.find(AUD_PATTERN, idx)
            if idx < 0:
                break
            candidate = pos + idx
            avail = block_len - idx
            if avail >= 11:
                # Validate from block buffer (zero extra I/O)
                if _validate_video(block, idx, avail, candidate, end):
                    return candidate
            else:
                # Near block boundary — read from file
                f.seek(candidate)
                tail = f.read(min(64, end - candidate))
                if len(tail) >= 11 and _validate_video(tail, 0, len(tail), candidate, end):
                    return candidate
            idx += 1

        pos += block_len - len(AUD_PATTERN) + 1

    return None


# ─── Audio chunk detection ───────────────────────────────────────────────────


def _bootstrap_audio_ref(f, audio_samples, audio_chunks, audio_ref, spc):
    """Bootstrap audio reference stats from equal-split samples.

    Reads the first 3 bytes of each audio frame from the first chunks to
    extract dominant max_sfb and ms_mask patterns, plus frame size stats.
    """
    all_sizes = [s for _, s in audio_samples]
    if not all_sizes:
        return

    # Frame size stats
    mean = sum(all_sizes) / len(all_sizes)
    variance = sum((s - mean) ** 2 for s in all_sizes) / len(all_sizes)
    audio_ref['frame_size_min'] = min(all_sizes)
    audio_ref['frame_size_max'] = max(all_sizes)
    audio_ref['frame_size_mean'] = mean
    audio_ref['frame_size_stdev'] = variance ** 0.5

    # Read first 3 bytes of each frame for header pattern detection
    msf_long_counts = {}
    msf_short_counts = {}
    ms_counts = {}

    for offset, sz in audio_samples:
        if sz < 3:
            continue
        f.seek(offset)
        hdr = f.read(3)
        if len(hdr) < 3:
            continue
        b0, b1, b2 = hdr[0], hdr[1], hdr[2]
        if b0 not in (0x20, 0x21) or (b1 & 0x80):
            continue
        ws = (b1 >> 5) & 0x03
        if ws == 2:
            msf = b1 & 0x0F
            if msf <= 14:
                msf_short_counts[msf] = msf_short_counts.get(msf, 0) + 1
        else:
            msf = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)
            if msf <= 49 and not (b2 & 0x20):
                msf_long_counts[msf] = msf_long_counts.get(msf, 0) + 1
        if b0 == 0x21 and ws != 2:
            ms = (b2 >> 3) & 0x03
            if ms != 3:
                ms_counts[ms] = ms_counts.get(ms, 0) + 1

    if msf_long_counts:
        audio_ref['dominant_msf_long'] = max(msf_long_counts,
                                              key=msf_long_counts.get)
    if msf_short_counts:
        audio_ref['dominant_msf_short'] = max(msf_short_counts,
                                               key=msf_short_counts.get)
    if ms_counts:
        audio_ref['dominant_ms'] = max(ms_counts, key=ms_counts.get)

def _create_audio_chunk(f, start, end, samples_per_chunk,
                        audio_samples, audio_chunks, audio_ref=None):
    """Split an audio region into samples using AAC frame boundary detection."""
    chunk_size = end - start
    if chunk_size < 10:
        return

    n = samples_per_chunk
    min_aac_frame = (audio_ref.get('frame_size_min') or 50) if audio_ref else 50
    if chunk_size < n * min_aac_frame:
        n = max(1, chunk_size // min_aac_frame)

    boundaries = None
    if n > 1:
        avg_frame = chunk_size / n
        if avg_frame <= 2000:
            f.seek(start)
            chunk_data = f.read(chunk_size)
            if len(chunk_data) == chunk_size:
                boundaries = _find_aac_boundaries(
                    chunk_data, n, audio_ref)

    chunk_sample_indices = []
    if boundaries and len(boundaries) == n + 1:
        for i in range(n):
            sz = boundaries[i + 1] - boundaries[i]
            if sz < 1:
                sz = 1
            audio_samples.append((start + boundaries[i], sz))
            chunk_sample_indices.append(len(audio_samples) - 1)
    else:
        # Fallback: equal splitting
        base_size = chunk_size // n
        remainder = chunk_size % n
        offset = start
        for i in range(n):
            sz = base_size + (1 if i < remainder else 0)
            audio_samples.append((offset, sz))
            chunk_sample_indices.append(len(audio_samples) - 1)
            offset += sz

    audio_chunks.append((start, chunk_sample_indices))


def _find_aac_boundaries(chunk_data, num_frames, audio_ref=None):
    """Find AAC-LC frame boundaries using DP minimum-variance partition.

    Uses candidate quality scoring and reference-weighted cost to prefer
    boundaries that produce frame sizes matching the known distribution.
    Tries tight reference-based bounds first, falls back to loose bounds.
    """
    ar = audio_ref or {}
    ref_min = ar.get('frame_size_min')
    ref_max = ar.get('frame_size_max')
    ref_mean = ar.get('frame_size_mean')
    ref_stdev = ar.get('frame_size_stdev')
    ref_msf_long = ar.get('dominant_msf_long')
    ref_msf_short = ar.get('dominant_msf_short')
    ref_ms = ar.get('dominant_ms')

    chunk_size = len(chunk_data)
    target_size = chunk_size / num_frames

    # Build candidates with quality scores
    # Score reflects how likely a candidate is a real CPE header
    candidates = [0]
    cand_scores = [0]  # position 0 is always valid (neutral score)

    for target_byte in (b'\x20', b'\x21'):
        is_cw1 = (target_byte == b'\x21')
        idx = 1
        while idx < chunk_size - 3:
            found = chunk_data.find(target_byte, idx, chunk_size - 3)
            if found < 0:
                break
            b1 = chunk_data[found + 1]
            b2 = chunk_data[found + 2]

            # ics_reserved_bit must be 0
            if b1 & 0x80:
                idx = found + 1
                continue

            ws = (b1 >> 5) & 0x03

            if ws == 2:  # EIGHT_SHORT: 4-bit max_sfb
                msf = b1 & 0x0F
                if msf > 14:
                    idx = found + 1
                    continue
            else:  # Long windows: 6-bit max_sfb + predictor must be 0
                msf = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)
                if msf > 49:
                    idx = found + 1
                    continue
                if b2 & 0x20:  # predictor_data_present must be 0
                    idx = found + 1
                    continue

            # Quality scoring
            score = 0
            if is_cw1:
                score += 3  # common_window=1 is dominant for stereo

            if ws == 0:  # ONLY_LONG most common (~90%+)
                score += 2
            elif ws == 2:  # EIGHT_SHORT is rare
                score -= 1

            # max_sfb matching: strongest discriminator
            if ws == 2:
                if ref_msf_short is not None and msf == ref_msf_short:
                    score += 6
            else:
                if ref_msf_long is not None:
                    if msf == ref_msf_long:
                        score += 6
                    elif msf == 0:
                        score += 1  # silence frames have msf=0
                    else:
                        score -= 3  # wrong msf is a strong negative signal

            # Check ms_mask_present for CW=1 long windows
            if is_cw1 and ws != 2:
                ms_mask = (b2 >> 3) & 0x03
                if ms_mask == 3:  # reserved value → reject
                    idx = found + 1
                    continue
                if ref_ms is not None:
                    if ms_mask == ref_ms:
                        score += 3
                    else:
                        score -= 1
                elif ms_mask <= 1:
                    score += 1

            candidates.append(found)
            cand_scores.append(score)
            idx = found + 1

    # Sort candidates (and scores together)
    if len(candidates) > 1:
        pairs = sorted(zip(candidates, cand_scores))
        candidates = [p[0] for p in pairs]
        cand_scores = [p[1] for p in pairs]
    nc = len(candidates)

    if nc < num_frames:
        return None

    # Safety cap: subsample to keep DP tractable, preserving high-quality candidates
    MAX_CANDIDATES = 500
    if nc > MAX_CANDIDATES:
        # Always keep index 0; keep all high-quality, subsample low-quality
        high_q = [i for i in range(1, nc) if cand_scores[i] >= 8]
        low_q = [i for i in range(1, nc) if cand_scores[i] < 8]
        budget = MAX_CANDIDATES - 1 - len(high_q)
        if budget > 0 and len(low_q) > budget:
            step = len(low_q) / budget
            low_q = [low_q[int(j * step)] for j in range(budget)]
        elif budget <= 0:
            low_q = []
        sampled_indices = sorted({0} | set(high_q) | set(low_q))
        candidates = [candidates[i] for i in sampled_indices]
        cand_scores = [cand_scores[i] for i in sampled_indices]
        nc = len(candidates)

    # Use reference stats for cost function
    cost_mean = ref_mean if ref_mean else target_size
    cost_scale = ref_stdev if ref_stdev and ref_stdev > 0 else max(1.0, target_size * 0.08)
    # Precompute for DP inner loop: inverse scale and pre-scaled quality bonuses
    inv_scale = 1.0 / cost_scale
    quality_weight = cost_scale * 0.6
    scaled_scores = [s * quality_weight for s in cand_scores]

    # Try tight bounds first, then loose
    bounds_list = []
    if ref_min is not None and ref_max is not None:
        bounds_list.append((max(1, int(ref_min * 0.7)),
                            int(ref_max * 1.3)))
    bounds_list.append((max(50, int(target_size * 0.15)),
                        int(target_size * 3.0)))

    for min_size, max_size in bounds_list:
        result = _dp_solve(candidates, scaled_scores, num_frames,
                           chunk_size, cost_mean, inv_scale,
                           min_size, max_size)
        if result is not None:
            return result

    return None


def _dp_solve(candidates, scaled_scores, num_frames, chunk_size,
              cost_mean, inv_scale, min_size, max_size):
    """DP partition with quality-weighted cost.

    scaled_scores: pre-multiplied quality bonuses per candidate.
    inv_scale: 1.0 / cost_scale for replacing division with multiplication.
    """
    nc = len(candidates)
    INF = float('inf')
    dp = [[INF] * nc for _ in range(num_frames)]
    parent = [[-1] * nc for _ in range(num_frames)]
    dp[0][0] = 0

    for fr in range(num_frames - 1):
        dp_fr = dp[fr]
        dp_next = dp[fr + 1]
        par_next = parent[fr + 1]
        for ci in range(nc):
            base = dp_fr[ci]
            if base >= INF:
                continue
            p = candidates[ci]
            lo = bisect_left(candidates, p + min_size, ci + 1)
            hi = bisect_right(candidates, p + max_size, lo)
            for cj in range(lo, hi):
                frame_size = candidates[cj] - p
                z = (frame_size - cost_mean) * inv_scale
                total = base + z * z - scaled_scores[cj]
                if total < dp_next[cj]:
                    dp_next[cj] = total
                    par_next[cj] = ci

    # Find best last-frame start
    best_cost = INF
    best_ci = -1
    dp_last = dp[num_frames - 1]
    for ci in range(nc):
        if dp_last[ci] >= INF:
            continue
        last_size = chunk_size - candidates[ci]
        if last_size < min_size or last_size > max_size:
            continue
        z = (last_size - cost_mean) * inv_scale
        total = dp_last[ci] + z * z
        if total < best_cost:
            best_cost = total
            best_ci = ci

    if best_ci == -1:
        return None

    # Backtrack
    path = [best_ci]
    for fr in range(num_frames - 1, 0, -1):
        path.append(parent[fr][path[-1]])
    path.reverse()

    return [candidates[ci] for ci in path] + [chunk_size]
