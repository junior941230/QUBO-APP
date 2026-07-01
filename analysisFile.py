import os
import mne
from collections import defaultdict
import pandas as pd

def scan_chb_channels(data_root):
    """
    掃描 CHB-MIT 資料夾下所有 EDF 檔案的 channel 資訊
    data_root: 資料集根目錄，例如 '/data/chb-mit'
    """
    
    results = []  # 儲存每個檔案的資訊
    channel_sets = defaultdict(list)  # 以 frozenset(channels) 為 key，記錄哪些檔案有這組 channels
    
    # 遞迴掃描所有 .edf 檔案
    edf_files = []
    for root, dirs, files in os.walk(data_root):
        for f in sorted(files):
            if f.endswith('.edf'):
                edf_files.append(os.path.join(root, f))
    
    print(f"找到 {len(edf_files)} 個 EDF 檔案，開始掃描...\n")
    
    for filepath in edf_files:
        filename = os.path.relpath(filepath, data_root)
        try:
            # 只讀 header，不載入資料（速度快）
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
            ch_names = raw.ch_names
            n_channels = len(ch_names)
            
            results.append({
                'file': filename,
                'n_channels': n_channels,
                'channels': ch_names
            })
            
            channel_sets[frozenset(ch_names)].append(filename)
            
        except Exception as e:
            print(f"  ❌ 無法讀取 {filename}: {e}")
            results.append({
                'file': filename,
                'n_channels': -1,
                'channels': []
            })
    
    return results, channel_sets


def report_channel_diff(results, channel_sets):
    """印出診斷報告"""
    
    print("=" * 60)
    print("📊 Channel 組合統計")
    print("=" * 60)
    
    # 找出最常見的 channel 組合（視為「標準」）
    standard_set = max(channel_sets.keys(), key=lambda k: len(channel_sets[k]))
    standard_channels = sorted(standard_set)
    
    print(f"\n✅ 標準 channel 組合（出現於 {len(channel_sets[standard_set])} 個檔案）：")
    print(f"   數量：{len(standard_channels)} channels")
    print(f"   名稱：{standard_channels}\n")
    
    # 列出所有不同的 channel 組合
    print(f"共有 {len(channel_sets)} 種不同的 channel 組合：\n")
    
    for i, (ch_set, files) in enumerate(
        sorted(channel_sets.items(), key=lambda x: -len(x[1]))
    ):
        ch_list = sorted(ch_set)
        print(f"  組合 #{i+1}：{len(ch_list)} channels，出現於 {len(files)} 個檔案")
        
        # 與標準組合比較差異
        extra = sorted(ch_set - standard_set)
        missing = sorted(standard_set - ch_set)
        
        if extra:
            print(f"    ➕ 多出的 channels：{extra}")
        if missing:
            print(f"    ➖ 缺少的 channels：{missing}")
        if not extra and not missing:
            print(f"    ✅ 與標準相同")
        
        # 列出屬於這個組合的檔案
        print(f"    📁 檔案：")
        for f in sorted(files)[:10]:  # 最多顯示 10 個
            print(f"       - {f}")
        if len(files) > 10:
            print(f"       ... 還有 {len(files)-10} 個檔案")
        print()
    
    print("=" * 60)
    print("⚠️  需要特別處理的檔案（非標準 channel 組合）：")
    print("=" * 60)
    
    non_standard = []
    for r in results:
        if r['n_channels'] != len(standard_channels) or \
           frozenset(r['channels']) != standard_set:
            non_standard.append(r)
    
    if non_standard:
        for r in non_standard:
            extra = sorted(set(r['channels']) - standard_set)
            missing = sorted(standard_set - set(r['channels']))
            print(f"\n  📄 {r['file']}")
            print(f"     Channel 數：{r['n_channels']}")
            if extra:
                print(f"     ➕ 多出：{extra}")
            if missing:
                print(f"     ➖ 缺少：{missing}")
    else:
        print("\n  🎉 所有檔案的 channel 組合一致！")
    
    return standard_channels, non_standard


# ── 執行 ──────────────────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = "DESTINATION/"  # ← 改成你的資料集路徑
    
    results, channel_sets = scan_chb_channels(DATA_ROOT)
    standard_channels, problem_files = report_channel_diff(results, channel_sets)
    
    # 可選：輸出成 CSV 方便查閱
    df = pd.DataFrame([{
        'file': r['file'],
        'n_channels': r['n_channels'],
        'channel_list': ', '.join(r['channels'])
    } for r in results])
    
    df.to_csv("chb_channel_audit.csv", index=False)
    print("\n📝 詳細結果已儲存至 chb_channel_audit.csv")
