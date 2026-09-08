"""Output file writer and ffmpeg audio fixer."""

import os
import re
import struct
import shutil
import subprocess


def write_output(corrupted_path, output_path, scan_info, moov):
    """Write recovered MP4: copy pre-mdat + mdat data + append moov."""
    mdat_offset = scan_info['mdat_offset']
    mdat_size = scan_info['mdat_size']

    with open(corrupted_path, 'rb') as src, open(output_path, 'wb') as dst:
        # Copy pre-mdat boxes (ftyp, uuid, etc.)
        src.seek(0)
        dst.write(src.read(mdat_offset))

        # Detect original mdat header size (8 standard, 16 extended)
        src.seek(mdat_offset)
        orig_size_field = struct.unpack('>I', src.read(4))[0]
        orig_hdr_size = 16 if orig_size_field == 1 else 8

        # Write mdat with correct size header.
        # data_bytes = actual payload bytes (mdat_size includes original header).
        data_bytes = mdat_size - orig_hdr_size
        if mdat_size > 0xFFFFFFFF:
            # Extended 64-bit size: total box = 16-byte header + data_bytes.
            dst.write(struct.pack('>I4sQ', 1, b'mdat', 16 + data_bytes))
        else:
            dst.write(struct.pack('>I4s', orig_hdr_size + data_bytes, b'mdat'))
        src.seek(mdat_offset + orig_hdr_size)
        remaining = data_bytes

        # Stream mdat content
        BUF = 4 * 1024 * 1024
        while remaining > 0:
            chunk = src.read(min(BUF, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)

        # Append moov
        dst.write(moov)


def fix_audio(output_path):
    """Fix audio: validate AAC stream, re-encode only if needed."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        print("  WARNING: ffmpeg not found - skipping audio fix.")
        print("  Run manually: ffmpeg -i OUTPUT -c:v copy -c:a aac -b:a 192k FIXED.mp4")
        return False

    # Check if audio decodes cleanly (lossless path)
    import time as _time
    print("  Validating audio stream...", end='', flush=True)
    t_chk = _time.perf_counter()
    r = subprocess.run(
        [ffmpeg, '-v', 'error', '-i', output_path, '-vn', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=600)
    aac_errors = [l for l in (r.stderr or '').split('\n')
                  if l.strip() and 'aac' in l.lower()]
    chk_elapsed = _time.perf_counter() - t_chk
    if not aac_errors:
        print(f" clean. ({chk_elapsed:.1f}s)")
        return True

    print(f" {len(aac_errors)} AAC errors. ({chk_elapsed:.1f}s)")
    print(f"  Re-encoding audio...")
    return _fix_audio_reencode(output_path, ffmpeg)


def _print_seg_progress(done, total, lossless, reencoded, silent, t0, action):
    """Print a single-line progress bar for segment processing."""
    import time as _time
    pct = 100 * done / total if total else 0
    filled = int(30 * done / total) if total else 0
    bar = '#' * filled + '-' * (30 - filled)
    elapsed = _time.perf_counter() - t0
    eta = (elapsed / done * (total - done)) if done > 0 else 0
    status = f"L:{lossless} R:{reencoded} S:{silent}"
    print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%) "
          f"{status} [{action}] "
          f"elapsed:{elapsed:.0f}s eta:{eta:.0f}s   ",
          end='', flush=True)


def _fix_audio_reencode(output_path, ffmpeg):
    """Hybrid audio fix: lossless copy for clean segments, re-encode corrupt ones.

    Optimised pipeline:
    1. Extract full audio track once (copy) — avoids repeated seeks into large file
    2. Extract + test segments in parallel from the small audio file
    3. Re-encode only bad segments in parallel
    4. Concatenate all segments and mux back with original video
    """
    import tempfile
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seg_dur = 2  # seconds per segment (small to maximize lossless)

    # Get total duration
    r = subprocess.run(
        [ffmpeg, '-i', output_path],
        capture_output=True, text=True, timeout=30)
    duration = 0.0
    for line in (r.stderr or '').split('\n'):
        m = re.search(r'Duration:\s*(\d+):(\d+):(\d+)\.(\d+)', line)
        if m:
            duration = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 +
                        int(m.group(3)) + int(m.group(4)) / 100)
            break
    if duration <= 0:
        print("  WARNING: Could not determine duration.")
        return False

    total_segs = int(duration / seg_dur) + (1 if duration % seg_dur else 0)
    workers = min(8, os.cpu_count() or 4)
    tmpdir = tempfile.mkdtemp(prefix='recover_mp4_')
    t0 = _time.perf_counter()

    try:
        # Phase 1: Extract full audio track (fast stream-copy from source)
        print("  Extracting audio track...", end='', flush=True)
        full_audio = os.path.join(tmpdir, 'full_audio.m4a')
        r = subprocess.run(
            [ffmpeg, '-y', '-v', 'error', '-i', output_path,
             '-vn', '-c:a', 'copy', full_audio],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f" failed.")
            return False
        print(" done.")

        # Phase 2: Extract + test segments in parallel (from small audio file)
        seg_files = [os.path.join(tmpdir, f'seg_{i:05d}.aac')
                     for i in range(total_segs)]

        def extract_and_test(i):
            """Extract segment from full_audio and test for AAC errors."""
            ss = i * seg_dur
            t = min(seg_dur, duration - ss)
            seg_path = seg_files[i]
            r = subprocess.run(
                [ffmpeg, '-y', '-v', 'error',
                 '-ss', f'{ss:.3f}', '-t', f'{t:.3f}',
                 '-i', full_audio, '-vn', '-c:a', 'copy',
                 '-f', 'adts', seg_path],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not os.path.exists(seg_path):
                return 'bad'
            r2 = subprocess.run(
                [ffmpeg, '-v', 'error', '-i', seg_path, '-f', 'null', '-'],
                capture_output=True, text=True, timeout=60)
            errs = [l for l in (r2.stderr or '').split('\n') if l.strip()]
            return 'bad' if errs else 'good'

        bad_indices = set()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_and_test, i): i
                       for i in range(total_segs)}
            for future in as_completed(futures):
                if future.result() == 'bad':
                    bad_indices.add(futures[future])
                done += 1
                if done % 20 == 0 or done == total_segs:
                    _print_seg_progress(done, total_segs,
                                        done - len(bad_indices), 0, 0,
                                        t0, 'testing')

        # Phase 3: Re-encode bad segments in parallel
        reencoded = 0
        silent = 0
        if bad_indices:
            print(f"\n  Re-encoding {len(bad_indices)} bad segments...",
                  end='', flush=True)

        def fix_seg(i):
            """Re-encode a bad segment; fall back to silence on failure."""
            ss = i * seg_dur
            t = min(seg_dur, duration - ss)
            seg_path = seg_files[i]
            tmp_path = seg_path + '.tmp'
            r = subprocess.run(
                [ffmpeg, '-y', '-v', 'error',
                 '-err_detect', 'ignore_err',
                 '-fflags', '+genpts',
                 '-ss', f'{ss:.3f}', '-t', f'{t:.3f}',
                 '-i', full_audio,
                 '-c:a', 'aac', '-ac', '2', '-b:a', '192k',
                 '-f', 'adts', tmp_path],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.exists(tmp_path):
                os.replace(tmp_path, seg_path)
                return 'reencoded'
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            subprocess.run(
                [ffmpeg, '-y', '-v', 'error',
                 '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
                 '-t', f'{t:.3f}',
                 '-c:a', 'aac', '-b:a', '192k', '-f', 'adts', seg_path],
                capture_output=True, text=True, timeout=30)
            return 'silent'

        if bad_indices:
            t_fix = _time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fix_seg, i): i for i in bad_indices}
                for future in as_completed(futures):
                    result = future.result()
                    if result == 'reencoded':
                        reencoded += 1
                    else:
                        silent += 1
            fix_elapsed = _time.perf_counter() - t_fix
            print(f" done. ({fix_elapsed:.1f}s)")

        lossless = total_segs - len(bad_indices)
        print(f"\r  Segments: {lossless}/{total_segs} lossless, "
              f"{reencoded}/{total_segs} re-encoded, "
              f"{silent}/{total_segs} silence.{' ' * 20}")

        # Phase 4: Concatenate ADTS segments (binary append — no ffmpeg concat
        # demuxer, which estimates file duration from size and introduces gaps)
        print("  Concatenating segments...", end='', flush=True)
        concat_audio = os.path.join(tmpdir, 'concat.aac')
        with open(concat_audio, 'wb') as out:
            for seg in seg_files:
                with open(seg, 'rb') as inp:
                    out.write(inp.read())
        print(" done.")

        # Phase 5: Mux original video + fixed audio
        print("  Muxing video + fixed audio...", end='', flush=True)
        tmp = output_path + '.tmp.mp4'
        r = subprocess.run(
            [ffmpeg, '-y', '-v', 'error',
             '-i', output_path, '-i', concat_audio,
             '-map', '0:v', '-map', '1:a',
             '-c:v', 'copy', '-c:a', 'copy',
             '-shortest', '-movflags', '+faststart', tmp],
            capture_output=True, text=True, timeout=3600)
        if r.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, output_path)
            elapsed = _time.perf_counter() - t0
            print(f" done. ({elapsed:.1f}s)")
            print("  Audio fixed.")
            return True
        print(f"\n  WARNING: Final mux failed: {(r.stderr or '')[-200:]}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmpdir, ignore_errors=True)
