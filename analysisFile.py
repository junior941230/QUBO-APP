import argparse
import os
from collections import defaultdict
from pathlib import Path

from core.channels import CANONICAL_CHB_CHANNELS, build_channel_plan


def _discover_edf_files(data_root):
    edf_files = []
    for root, _dirs, files in os.walk(data_root):
        for filename in sorted(files):
            if filename.lower().endswith(".edf"):
                edf_files.append(os.path.join(root, filename))
    return sorted(edf_files)


def scan_chb_channels(data_root, read_raw_edf=None):
    """Scan EDF headers and validate the canonical CHB bipolar montage.

    Returns one result per EDF plus a mapping of raw channel combinations to
    files. Compatible recordings may use direct bipolar channels or channels
    that can be reconstructed from a common reference.
    """
    if read_raw_edf is None:
        import mne

        read_raw_edf = mne.io.read_raw_edf

    data_root = os.fspath(data_root)
    results = []
    channel_sets = defaultdict(list)
    edf_files = _discover_edf_files(data_root)

    print(f"找到 {len(edf_files)} 個 EDF 檔案，開始驗證 channel header...\n")

    for filepath in edf_files:
        filename = os.path.relpath(filepath, data_root)
        raw = None
        try:
            raw = read_raw_edf(filepath, preload=False, verbose=False)
            ch_names = list(raw.ch_names)
            channel_sets[frozenset(ch_names)].append(filename)

            try:
                plan = build_channel_plan(ch_names)
            except Exception as exc:
                result = {
                    "file": filename,
                    "n_channels": len(ch_names),
                    "channels": ch_names,
                    "status": "incompatible",
                    "compatible": False,
                    "validation_error": str(exc),
                    "direct_channels": 0,
                    "reconstructed_channels": 0,
                }
            else:
                direct_count = sum(item[0] == "direct" for item in plan)
                result = {
                    "file": filename,
                    "n_channels": len(ch_names),
                    "channels": ch_names,
                    "status": "compatible",
                    "compatible": True,
                    "validation_error": "",
                    "direct_channels": direct_count,
                    "reconstructed_channels": len(plan) - direct_count,
                }
        except Exception as exc:
            print(f"  ❌ 無法讀取 {filename}: {exc}")
            result = {
                "file": filename,
                "n_channels": -1,
                "channels": [],
                "status": "unreadable",
                "compatible": False,
                "validation_error": str(exc),
                "direct_channels": 0,
                "reconstructed_channels": 0,
            }
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        results.append(result)

    return results, channel_sets


def report_channel_diff(results, channel_sets):
    """Print raw channel variants and canonical-montage validation results."""
    print("=" * 60)
    print("📊 Channel 組合統計")
    print("=" * 60)

    if not channel_sets:
        print("\n❌ 沒有可讀取的 EDF channel header。")
        return [], list(results)

    standard_set = max(channel_sets, key=lambda key: len(channel_sets[key]))
    standard_channels = sorted(standard_set)

    print(f"\n最常見的原始 channel 組合（{len(channel_sets[standard_set])} 個檔案）：")
    print(f"   數量：{len(standard_channels)} channels")
    print(f"   名稱：{standard_channels}\n")
    print(f"共有 {len(channel_sets)} 種不同的原始 channel 組合：\n")

    for index, (channel_set, files) in enumerate(
        sorted(channel_sets.items(), key=lambda item: -len(item[1])),
        start=1,
    ):
        channel_list = sorted(channel_set)
        print(
            f"  組合 #{index}：{len(channel_list)} channels，"
            f"出現於 {len(files)} 個檔案"
        )

        extra = sorted(channel_set - standard_set)
        missing = sorted(standard_set - channel_set)
        if extra:
            print(f"    ➕ 相較最常見組合多出：{extra}")
        if missing:
            print(f"    ➖ 相較最常見組合缺少：{missing}")
        if not extra and not missing:
            print("    ✅ 與最常見組合相同")

        print("    📁 檔案：")
        for filename in sorted(files)[:10]:
            print(f"       - {filename}")
        if len(files) > 10:
            print(f"       ... 還有 {len(files) - 10} 個檔案")
        print()

    non_standard = [
        result
        for result in results
        if result["n_channels"] != len(standard_channels)
        or frozenset(result["channels"]) != standard_set
    ]

    compatible = [result for result in results if result["compatible"]]
    failures = [result for result in results if not result["compatible"]]
    reconstructed = [
        result for result in compatible if result["reconstructed_channels"] > 0
    ]

    print("=" * 60)
    print("🧪 Canonical 18-channel montage 驗證")
    print("=" * 60)
    print(f"Canonical channels：{list(CANONICAL_CHB_CHANNELS)}")
    print(f"✅ 通過：{len(compatible)}")
    print(f"🔧 其中需要重建 channel：{len(reconstructed)}")
    print(f"❌ 未通過：{len(failures)}")

    if reconstructed:
        print("\n需要由 common-reference 重建的檔案：")
        for result in reconstructed:
            print(
                f"  - {result['file']}: "
                f"{result['reconstructed_channels']} channels"
            )

    if failures:
        print("\n未通過驗證的檔案：")
        for result in failures:
            print(
                f"  - {result['file']} [{result['status']}]: "
                f"{result['validation_error']}"
            )
    else:
        print("\n🎉 所有 EDF 都能建立固定的 canonical montage。")

    return standard_channels, non_standard


def write_channel_audit_csv(results, output_path):
    """Write the validation details to CSV."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([
        {
            "file": result["file"],
            "status": result["status"],
            "compatible": result["compatible"],
            "validation_error": result["validation_error"],
            "n_channels": result["n_channels"],
            "direct_channels": result["direct_channels"],
            "reconstructed_channels": result["reconstructed_channels"],
            "channel_list": ", ".join(result["channels"]),
        }
        for result in results
    ])
    dataframe.to_csv(output_path, index=False)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate CHB-MIT EDF channels before running experiments."
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default="DESTINATION",
        help="Dataset root containing subject folders (default: DESTINATION).",
    )
    parser.add_argument(
        "--output",
        default="chb_channel_audit.csv",
        help="CSV report path (default: chb_channel_audit.csv).",
    )
    args = parser.parse_args(argv)

    results, channel_sets = scan_chb_channels(args.data_root)
    report_channel_diff(results, channel_sets)
    output_path = write_channel_audit_csv(results, args.output)
    print(f"\n📝 詳細結果已儲存至 {output_path}")

    failures = [result for result in results if not result["compatible"]]
    if not results or failures:
        print("\n❌ Channel preflight 未通過，請先處理上述檔案。")
        return 1

    print("\n✅ Channel preflight 通過，可以開始執行實驗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
