import struct
import tempfile
import unittest
from pathlib import Path

from atoms import build_stsd_audio
from x264_recovery import (
    DEFAULT_CONFIG,
    X264Config,
    _candidate,
    _infer_audio_rate,
    _mdat_info,
    _parse_x264_options,
    _slice_header,
)


def _ue(value):
    code = format(value + 1, 'b')
    return '0' * (len(code) - 1) + code


def _bits_to_bytes(bits):
    bits += '1'  # rbsp_stop_one_bit
    bits += '0' * ((8 - len(bits) % 8) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _length_prefixed_slice(slice_type, frame_num=0, nal_type=1,
                           size=None, frame_num_bits=4):
    bits = _ue(0) + _ue(slice_type) + _ue(0)
    bits += format(frame_num, f'0{frame_num_bits}b')
    payload = _bits_to_bytes(bits)
    if size is not None:
        payload += b'\x00' * max(0, size - 1 - len(payload))
    nal = bytes([0x41 if nal_type == 1 else 0x65]) + payload
    return struct.pack('>I', len(nal)) + nal


class X264RecoveryTests(unittest.TestCase):
    def test_defaults_describe_investigated_recording(self):
        self.assertEqual(DEFAULT_CONFIG.video_timescale, 60000)
        self.assertEqual(DEFAULT_CONFIG.video_delta, 1000)
        self.assertEqual(DEFAULT_CONFIG.audio_rate, 48000)

    def test_x264_option_parser_does_not_confuse_prefixed_names(self):
        data = (b'x264 - core 164 - options: cabac=1 ref=3 '
                b'mixed_ref=1 bframes=3 keyint_min=59\x00')
        self.assertEqual(_parse_x264_options(data), {
            'cabac': 1, 'ref': 3, 'bframes': 3, 'keyint_min': 59,
        })

    def test_interleave_rate_inference_selects_nearest_standard_rate(self):
        self.assertEqual(_infer_audio_rate(218775, 171098, DEFAULT_CONFIG), 48000)

    def test_slice_header_reads_configured_frame_number_width(self):
        packet = _length_prefixed_slice(6, frame_num=17, size=8,
                                       frame_num_bits=5)
        self.assertEqual(_slice_header(packet, 5, len(packet) - 4, 5),
                         (0, 6, 0, 17))

    def test_candidate_accepts_x264_modulo_slice_types(self):
        view = _length_prefixed_slice(6, frame_num=3, size=8)
        self.assertEqual(_candidate(view, 0, len(view))[2][1], 6)

    def test_candidate_rejects_syntax_valid_but_unseen_slice_type(self):
        # H.264 permits slice_type 0, but this x264 stream writes 5/6/7.
        view = _length_prefixed_slice(0, frame_num=3)
        self.assertIsNone(_candidate(view, 0, len(view)))

    def test_candidate_rejects_tiny_type5_false_positive(self):
        view = _length_prefixed_slice(7, nal_type=5)
        self.assertIsNone(_candidate(view, 0, len(view)))

    def test_candidate_accepts_realistic_type5_idr(self):
        view = _length_prefixed_slice(7, nal_type=5, size=10_000)
        self.assertIsNotNone(_candidate(view, 0, len(view)))

    def test_mdat_info_handles_size_zero_box(self):
        payload = b'raw media'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'damaged.mp4'
            path.write_bytes(
                struct.pack('>I4s', 8, b'ftyp') +
                struct.pack('>I4s', 0, b'mdat') + payload)
            self.assertEqual(_mdat_info(str(path)),
                             (8, 8 + len(payload), 16, 16 + len(payload)))

    def test_audio_stsd_uses_requested_aac_config(self):
        stsd = build_stsd_audio(44100, channels=1)
        self.assertIn(b'\x12\x08', stsd)  # AAC-LC, 44.1 kHz, mono
        mp4a = stsd.index(b'mp4a')
        self.assertEqual(struct.unpack_from('>H', stsd, mp4a + 20)[0], 1)

    def test_custom_recording_settings_are_representable(self):
        config = X264Config(width=2560, height=1440, fps_num=30000,
                            fps_den=1001, audio_rate=44100,
                            audio_channels=1)
        self.assertEqual(config.video_timescale, 30_000_000)
        self.assertEqual(config.video_delta, 1001_000)


if __name__ == '__main__':
    unittest.main()
