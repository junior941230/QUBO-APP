import unittest

from core.channels import (
    CANONICAL_CHB_CHANNELS,
    build_channel_plan,
    validate_edf_channels,
)


class ChannelPlanTests(unittest.TestCase):
    def test_bipolar_montage_uses_t8_p8_zero_alias(self):
        channels = [
            "T8-P8-0" if name == "T8-P8" else name
            for name in CANONICAL_CHB_CHANNELS
        ]
        channels.extend(["T8-P8-1", "--0"])

        plan = build_channel_plan(channels)

        self.assertEqual(len(plan), 18)
        self.assertEqual(plan[16], ("direct", "T8-P8-0"))

    def test_common_reference_montage_is_reconstructed(self):
        electrodes = {
            electrode
            for channel in CANONICAL_CHB_CHANNELS
            for electrode in channel.split("-")
        }
        channels = [f"{name}-CS2" for name in electrodes]

        plan = build_channel_plan(channels)

        self.assertEqual(len(plan), 18)
        self.assertTrue(all(item[0] == "difference" for item in plan))

    def test_plain_electrodes_accept_zero_one_alias(self):
        electrodes = {
            electrode
            for channel in CANONICAL_CHB_CHANNELS
            for electrode in channel.split("-")
        }
        channels = ["01" if name == "O1" else name for name in electrodes]

        plan = build_channel_plan(channels)

        p7_o1 = plan[3]
        self.assertEqual(p7_o1, ("difference", "P7", "01"))

    def test_incomplete_montage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing channels"):
            build_channel_plan(["FP1-F7"])

    def test_edf_preflight_excludes_incompatible_and_unreadable_files(self):
        channel_sets = {
            "good.edf": list(CANONICAL_CHB_CHANNELS),
            "missing.edf": ["FP1-F7"],
        }
        opened = []

        class FakeRaw:
            def __init__(self, path):
                self.ch_names = channel_sets[path]
                self.closed = False

            def close(self):
                self.closed = True

        def fake_read_raw_edf(path, preload, verbose):
            self.assertFalse(preload)
            self.assertFalse(verbose)
            if path == "unreadable.edf":
                raise OSError("invalid EDF header")
            raw = FakeRaw(path)
            opened.append(raw)
            return raw

        valid, failures = validate_edf_channels(
            ["good.edf", "missing.edf", "unreadable.edf"],
            read_raw_edf=fake_read_raw_edf,
        )

        self.assertEqual(valid, ["good.edf"])
        self.assertIn("missing channels", failures["missing.edf"])
        self.assertEqual(failures["unreadable.edf"], "invalid EDF header")
        self.assertTrue(all(raw.closed for raw in opened))


if __name__ == "__main__":
    unittest.main()
