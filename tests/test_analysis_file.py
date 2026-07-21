import tempfile
import unittest
from pathlib import Path

from analysisFile import scan_chb_channels
from core.channels import CANONICAL_CHB_CHANNELS


class ChannelAuditTests(unittest.TestCase):
    def test_scan_classifies_compatible_incompatible_and_unreadable_edfs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for filename in ("good.edf", "missing.edf", "unreadable.edf"):
                (root / filename).touch()

            opened = []

            class FakeRaw:
                def __init__(self, channel_names):
                    self.ch_names = channel_names
                    self.closed = False

                def close(self):
                    self.closed = True

            def fake_read_raw_edf(path, preload, verbose):
                self.assertFalse(preload)
                self.assertFalse(verbose)
                filename = Path(path).name
                if filename == "unreadable.edf":
                    raise OSError("bad header")
                channels = (
                    list(CANONICAL_CHB_CHANNELS)
                    if filename == "good.edf"
                    else ["FP1-F7"]
                )
                raw = FakeRaw(channels)
                opened.append(raw)
                return raw

            results, channel_sets = scan_chb_channels(
                root,
                read_raw_edf=fake_read_raw_edf,
            )

        by_file = {result["file"]: result for result in results}
        self.assertEqual(by_file["good.edf"]["status"], "compatible")
        self.assertEqual(by_file["missing.edf"]["status"], "incompatible")
        self.assertIn("missing channels", by_file["missing.edf"]["validation_error"])
        self.assertEqual(by_file["unreadable.edf"]["status"], "unreadable")
        self.assertEqual(len(channel_sets), 2)
        self.assertTrue(all(raw.closed for raw in opened))


if __name__ == "__main__":
    unittest.main()
