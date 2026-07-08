import unittest

from core.channels import CANONICAL_CHB_CHANNELS, build_channel_plan


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


if __name__ == "__main__":
    unittest.main()
