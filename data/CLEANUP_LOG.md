# データディレクトリ クリーンアップログ

## 現在のディレクトリ構成（2024年8月）

### 📁 必要なファイル（保持）
- `playdata/` - 実際の演奏データ（2,433セッション）
- `all_answer_summary_df.csv` - プロ/アマラベル（6,476サンプル）
- `README.md` - データセット説明書
- `full_splits/` - 学習/検証/テスト分割
- `preparation_report.json` - データ準備レポート

### 📦 アーカイブファイル（展開済みのため削除候補）
- `2024skillcheck_playdata.tar.xz` (9.2GB) - playdataディレクトリに展開済み
- `data.zip` (9.2GB) - 同じデータの別形式アーカイブ

### 📊 その他のファイル
- `2024skillcheck_scale_arpeggio_unrefined_raw.csv` - スケール/アルペジオの生データ
- `._playdata` - macOSのメタデータファイル（削除可能）

## 推奨アクション

1. **アーカイブファイルの削除**（18.4GB節約）
   - 既にplaydataディレクトリに展開済みのため不要
   - 必要に応じて外部ストレージにバックアップ

2. **システムファイルの削除**
   - `._playdata` - macOSの隠しファイル

## ストレージ使用状況
- 現在: 約19GB
- クリーンアップ後: 約600MB（playdataディレクトリのみ）

## 注意事項
- playdataディレクトリは実験に必須のため絶対に削除しないこと
- all_answer_summary_df.csvはラベルファイルのため必須
- アーカイブファイルは削除前に必ずバックアップを検討すること