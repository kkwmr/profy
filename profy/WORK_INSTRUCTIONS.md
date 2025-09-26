# 作業指示書

## 🚨 最重要要件

### ⚠️⚠️⚠️ 絶対要件：ALL 6,476サンプル使用 ⚠️⚠️⚠️
- **必ずALL 6,476サンプルを使用** (Pro: 3,782, Amateur: 2,694)
- **`load_real_piano_data(max_samples=None)`を必ず使用**
- **max_samples=2000等の制限は絶対禁止**

### 🚫🚫🚫 シミュレーション・偽データ使用絶対禁止 🚫🚫🚫
- **torch.randn()等のランダムデータ生成は絶対禁止**
- **必ず実際のWAVファイルから音声データを読み込むこと**
- **シミュレーションデータの使用は一切認めない**
- **偽データを使用した場合、実験結果は無効**

## 📌 統一要件

### 1. 統一モデル（UnifiedAttentionModel）
- **使用モデル**: `src/models/unified_attention_model.py`のみ
- **入力対応**: センサーデータと音声データの両方を入力可能
- **F1ベースライン**: 0.670±0.072

### 2. 3モダリティ比較実験
必ず以下の3パターンで比較：
1. **センサーのみ** (model(sensor_data, None))
2. **音声のみ** (model(None, audio_data))
3. **マルチモーダル** (model(sensor_data, audio_data))

### 3. Player-wise GroupKFold
- **必須**: GroupKFoldによる演奏者単位の分割
- **検証**: 同一演奏者がtrain/testに重複しないこと
- **実装**: `session_ids`をgroupsパラメータに使用

### 4. 時系列Attention可視化
- **目的**: アマチュアのどこがプロっぽくないか時系列で可視化
- **出力**: attentionの時間軸プロット
- **保存形式**: PNG/PDF

### 5. 楽曲ごとのヒートマップ
- **形式**: 縦軸=3モダリティ、横軸=楽曲
- **メトリクス**: F1スコア
- **カラーマップ**: 性能の高低を色で表現

## 🗂️ プロジェクト構造（整理済み）

```
profy/
├── src/
│   ├── models/
│   │   └── unified_attention_model.py  # ★統一モデル（これのみ使用）
│   ├── data/
│   │   └── real_data_loader.py        # ★実データ読み込み（メイン使用）
│   ├── training/                      # 参考用（run.shで全機能実装済み）
│   │   ├── train_multimodal_comparison.py
│   │   ├── train_simple_comparison.py
│   │   ├── train_with_batches.py
│   │   └── train_with_player_split.py
│   ├── evaluation/                    # 評価関連（必要時使用）
│   ├── utils/                         # ユーティリティ
│   └── visualization/                 # 可視化ツール
├── scripts/
│   └── run.sh                         # ★メイン実行スクリプト（これのみ使用）
├── results/                           # 実験結果（最新のみ保持）
└── WORK_INSTRUCTIONS.md               # この指示書
```

## 🚨 シェルスクリプト管理規則

### 重要: scripts/run.sh のみ使用
- **scripts/run.sh**: メインの実験実行スクリプト（最新状態を維持）
- **他のshファイル作成禁止**: run.sh以外のシェルスクリプトは作成しない
- **更新時は run.sh のみ修正**: 新機能追加時もrun.shを更新すること
- **重複防止**: run_*.sh等の派生スクリプトを作らない

## 📊 実データ仕様

### センサーデータ
- **パス**: `/home/kazuki/Projects/Profy/data/playdata/{id}/files/hackkey/hackkey.csv`
- **形式**: 88鍵盤×1000Hz
- **前処理**: 500サンプルごとに平均化

### 音声データ
- **パス**: `/home/kazuki/Projects/Profy/data/playdata/{id}/files/sound/*.wav`
- **特徴**: MFCCs 40次元 + スペクトル特徴 3次元

### メタデータ
- **ファイル**: `/home/kazuki/Projects/Profy/data/all_answer_summary_df.csv`
- **ラベル**: player_tag (pro/amateur)

## ⚙️ 実験実行手順

### 1. データ読み込み（約17分）
```python
X, y, metadata = load_real_piano_data(max_samples=None)  # ALL 6,476
```

### 2. GroupKFold設定
```python
session_ids = np.array([m['session_id'] for m in metadata])
gkf = GroupKFold(n_splits=3)
```

### 3. バッチ処理（GPU対策）
- **バッチサイズ**: 32サンプル
- **理由**: GPU OOMエラー回避

### 4. 3モダリティ実行
各fold、各modalityで学習・評価を実施

### 5. 結果保存
- 性能メトリクス (JSON)
- Attentionプロット (PNG)
- 楽曲ヒートマップ (PNG)

## 🧹 クリーンアップ要件

### 削除対象
- 古い実験結果（最新のみ保持）
- 重複するスクリプト
- 不要なログファイル

### 保持対象
- 最新の実験結果
- UnifiedAttentionModel関連
- 実データローダー

## ⚠️ 注意事項

### GPU関連
- **問題**: CUDA OOMエラー頻発
- **対策**: バッチ処理必須（32サンプル/バッチ）
- **メモリ解放**: 各fold後に`torch.cuda.empty_cache()`

### データ読み込み
- **時間**: 約17-20分/回
- **対策**: キャッシュ検討（data_cache.npz）

### 現在の課題
1. ✅ 音声データ実装完了（load_multimodal_dataで実WAVファイル読み込み）
2. 複数の重複スクリプト存在
3. Attention可視化の完全実装待ち

## 📝 チェックリスト

実験実行前に必ず確認：
- [ ] ALL 6,476サンプル使用しているか
- [ ] UnifiedAttentionModel使用か
- [ ] GroupKFold実装されているか
- [ ] 3モダリティ比較があるか
- [ ] バッチ処理（32サンプル）か
- [ ] 結果保存先が明確か
- [ ] 不要ファイル削除したか

## 🎯 最終目標

1. **分類性能**: プロ/アマを高精度で分類
2. **解釈性**: アマチュアの問題箇所を時系列で特定
3. **比較**: 3モダリティの性能を定量評価
4. **可視化**: 楽曲別性能をヒートマップで表示
5. **再現性**: 単一スクリプトで完全再現可能