import os
import pandas as pd
import shutil

# データの読み込み
file_path = os.path.join('..','250624_af_dataset_all.xlsx')
wav_af_df = pd.read_excel(file_path)

# データ数を確認
print(f"データ数: {len(wav_af_df)}")

# パスの設定
wav_all_path = os.path.join('..', '..','wav/wav')
wav_zch_path = os.path.join('..', '..','wav/ZCHSound/clean Heartsound Data')
wav_af_path = os.path.join('..', '..','wav_af')

# wav_af_pathフォルダが存在しない場合は作成
os.makedirs(wav_af_path, exist_ok=True)

# ZCHSound列が1の場合にwav_zch_pathからファイルをコピーする処理
copied_files = []
missing_files = []
zch_copied = 0
wav_all_copied = 0

for index, row in wav_af_df.iterrows():
    wav_name = row['wav_name']
    
    # ZCHSound列が1の場合はwav_zch_pathから、そうでなければwav_all_pathからコピー
    if 'ZCHSound' in row and row['ZCHSound'] == 1:
        source_file = os.path.join(wav_zch_path, wav_name)
        source_type = "ZCHSound"
    else:
        source_file = os.path.join(wav_all_path, wav_name)
        source_type = "wav_all"
    
    dest_file = os.path.join(wav_af_path, wav_name)
    
    if os.path.exists(source_file):
        try:
            shutil.copy2(source_file, dest_file)
            copied_files.append(wav_name)
            if source_type == "ZCHSound":
                zch_copied += 1
            else:
                wav_all_copied += 1
            print(f"コピー完了 ({source_type}): {wav_name}")
        except Exception as e:
            print(f"コピーエラー {wav_name}: {e}")
    else:
        missing_files.append(wav_name)
        print(f"ファイルが見つかりません ({source_type}): {wav_name}")

print(f"\n総計: {len(copied_files)}ファイルをコピーしました")
print(f"  - ZCHSoundから: {zch_copied}ファイル")
print(f"  - wav_allから: {wav_all_copied}ファイル")
print(f"見つからないファイル: {len(missing_files)}個")

# 統計情報の表示
if 'ZCHSound' in wav_af_df.columns:
    zch_count = (wav_af_df['ZCHSound'] == 1).sum()
    print(f"\nZCHSound列が1のレコード数: {zch_count}")
    print(f"ZCHSound列が0またはNaNのレコード数: {len(wav_af_df) - zch_count}")
else:
    print("\nZCHSound列がデータフレームに存在しません") 