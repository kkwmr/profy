# Profy: AIによるピアノ演奏評価・指導システム

**Profy** は、MERT（Music-understanding Efficient pre-trained Transformer）と物理センサーデータを活用して、ピアノ演奏の詳細な分析と具体的な改善指導を提供するAIコーチングシステムです。

## 🎯 プロジェクトの目標

### 現在の改善状況
- **Phase 1 (完了)**: モデル学習の基本的な問題を修正
  - ✅ 分類器バイアスの初期化修正（少数クラスに適切な初期値）
  - ✅ Focal Lossによるクラス不均衡対策（α=0.55/0.45, γ=1.0）
  - ✅ 学習率の最適化（5e-5）とCosine Annealingスケジューラ
  
- **Phase 2 (完了)**: データパイプラインの改善
  - ✅ 音声（24kHz）・センサー（1000Hz）データの厳密な時間同期
  - ✅ note.jsonデータからの音符レベル特徴抽出（NoteFeatureExtractor）
  - ✅ 音楽的評価メトリクス（タイミング精度、ダイナミクス一貫性、レガート品質）

### 🎼 システムの特徴
- **マルチモーダル統合分析**: 音声（24kHz）と88鍵センサーデータ（1000Hz）の同期処理
- **物理的演奏解析**: キー押下深度、速度、加速度の詳細測定（mm単位）
- **具体的な改善指導**: どの音符で、どのような技術的問題があるかを特定

### 🧠 改良されたAIアーキテクチャ
- **階層的MERT特徴抽出**: 複数層からの特徴を統合利用
- **時間同期メカニズム**: 音声とセンサーデータの正確な対応付け
- **Physics-Informed設計**: 物理的制約を考慮した学習

## 🚀 クイックスタート

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-username/profy.git
cd profy

# conda環境の作成
conda create -n profy python=3.9 -y
conda activate profy

# 依存関係のインストール
pip install -r requirements.txt
```

### 実行方法

```bash
# データ準備
python scripts/prepare_data.py

# モデル訓練（改善されたconfig使用）
python scripts/train.py --config configs/config.yaml --output-dir results/$(date +%Y%m%d_%H%M%S)_improved --data-dir data/full_splits

# モデル評価
python scripts/evaluate.py --model results/MODEL_DIR/best_model.pth --data-dir data/full_splits --output-dir results/evaluation

# 結果の可視化
python scripts/visualize.py
```

## 📁 プロジェクト構成

```
profy/
├── configs/            # ⚙️ 設定ファイル
│   └── config.yaml     # 基本設定
├── src/               # 💻 ソースコード
│   ├── models/        # ニューラルネットワークモデル
│   ├── data/          # データ処理
│   ├── evaluation/    # 評価関連
│   └── visualization/ # 可視化ツール
├── scripts/           # 🔧 実行スクリプト
│   ├── train.py       # モデル訓練
│   ├── evaluate.py    # モデル評価
│   ├── prepare_data.py # データ準備
│   └── visualize.py   # 結果可視化
├── data/              # 📊 入力データ
│   ├── playdata/      # 演奏データ
│   ├── full_splits/   # 訓練/検証/テストデータ
│   └── all_answer_summary_df.csv # 評価スコア
├── results/           # 📈 実験結果
│   └── YYYYMMDD_HHMMSS_training/ # 各訓練セッションの結果
├── logs/              # 📝 ログファイル（現在は空）
├── README.md          # 📖 このファイル
├── WORK_INSTRUCTIONS.md # 📋 開発ガイドライン
└── requirements.txt   # 📦 Python依存関係
```

### ディレクトリ管理ルール

- **logs/**: すべてのログファイルはこのディレクトリに保存
- **results/**: 実験結果は日時付きのサブディレクトリに整理
- **ルートディレクトリ**: ログファイルを直接配置しない

## 🔧 実行の流れ

1. **データ準備** (`prepare_data.py`)
   - 音声ファイルとセンサーデータの読み込み
   - 訓練/検証/テストセットへの分割（70%/15%/15%）
   - メタデータの生成

2. **モデル訓練** (`train.py`)
   - MERT-v1-330Mベースのマルチモーダルモデルの訓練
   - 複数タスクの同時学習（スコア分類、技術的特徴の回帰）
   - 最良モデルの自動保存

3. **モデル評価** (`evaluate.py`)
   - テストセットでの性能評価
   - 精度、F1スコア、平均絶対誤差の計算
   - Attention統計の分析

4. **可視化生成** (`visualize.py`)
   - 複合可視化（音声波形＋センサーデータ＋Attention）
   - Attention重みのプロットとヒートマップ
   - インタラクティブHTMLレポート

## 📊 データセット概要

### データ収集の概要
- **有効セッション数**: 237セッション（メタデータ付き544セッション中）
- **評価レコード数**: 6,517件の専門家評価

### データモダリティ

#### 1. **音声データ** 🎵
- `sound_raw.wav`: 演奏音声（16kHz、モノラル）

#### 2. **センサーデータ** 🎹
- `hackkey.csv`: 88鍵盤の押下データ
  - サンプリングレート: 1000Hz（1ms解像度）
  - 値: キー押下深度（0-127）

#### 3. **評価スコア** 📝
- `score1`: 主要評価（0-6スケール）
- ラベル生成: score1 ≥ 4 を「良い演奏」として二値分類

## 🔧 設定ファイル

### 基本設定 (`configs/config.yaml`)
- MERT-v1-330Mを使用した標準設定
- マルチタスク学習（スコア分類＋技術評価）
- 200エポックの訓練

## 📋 開発ガイドライン

### ファイル管理のベストプラクティス

#### ✅ 推奨される対応
- **既存ファイルの修正を優先** - 新規ファイル作成前に既存ファイルの改良を検討
- **機能分割** - 複雑になった場合のみファイルを分割
- **一時ファイルの削除** - 実験やテスト用ファイルは使用後に必ず削除
- **可読性重視** - コードの整理と適切なコメント

#### ❌ 避けるべき対応
- **不要なファイル作成** - 既存ファイルで対応可能な場合の新規作成
- **一時ファイルの放置** - `test_*.py`, `debug_*.py`, `temp_*.txt`などの残存
- **構造の複雑化** - 必要以上のディレクトリ階層やファイル分割

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- **データセット**: ピアノ演奏データセットの提供元
- **モデル**: Hugging Face Transformersに基づく実装
- **インスピレーション**: 言語学習向けDDSupportの方法論

---

**Profy** - *AIの力でピアノ演奏を向上させる* 🎹✨
# Profy: Multimodal Piano Performance Analysis and Highlights

This repository contains the Profy system for learning expertise‑dependent differences from synchronized key‑sensor (1 kHz) and audio streams and rendering localized, score‑synchronized highlights that guide practice.

## Model Overview

- Backbone: `UnifiedAttentionModel` (see `src/models/unified_attention_model.py`)
  - Sensor encoder: 1D CNN with multi‑kernel fusion (3/5/9), max‑pooling, projection to hidden.
  - Audio encoder: 2D CNN over time–frequency with a frequency‑attention head that learns per‑band weights before temporal pooling (prevents losing informative bands).
  - Cross‑modal attention: bidirectional multi‑head attention (sensor↔audio) with key‑padding masks; parametric resamplers align streams to a common temporal grid.
  - Fusion: quality‑aware gating (uses audio quality vector: non‑silence rate, spectral flatness, loudness) and temperature scaling; outputs per‑sample modality weights and entropy.
  - Temporal modeling: BiLSTM → temporal attention (with masking) and an evidence head (sigmoid per frame) + a global classifier (expert vs amateur).
  - Masking: audio/sensor masks propagate to attention/evidence to suppress silent/invalid frames.

## Training Losses

- Classification: cross‑entropy on global logits.
- Evidence learning (weak supervision):
  - MIL (Noisy‑OR) over frames to match clip‑level label.
  - Soft Top‑k pooling (mean of top `k = frac * L` frames) blended with Noisy‑OR (`alpha`), stabilizing against spiky single‑frame peaks.
  - L1 sparsity on evidence to encourage localized peaks.
- Attention energy regularization (optional): discourages attention aligning purely with frame energy (approx. mean‑abs over 128 audio features).

Defaults (debug/full): `--lambda-mil 0.5`, `--lambda-evidence-l1 1e-3`, `--mil-topk-frac 0.1`, `--mil-blend-alpha 0.5`, `--lambda-attention-energy 0.0` (set to small positive e.g. `0.002` to enable).

## Reproducible Runs

### 1) Environment

- Python 3.10 recommended. Activate venv and install web UI deps as needed:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r webapp/requirements.txt` (only if running the local web app)

### 2) End‑to‑End Experiment (Debug)

Runs a quick pipeline (≈60 samples) to verify everything:

```
./scripts/run.sh --debug --lambda-attention-energy 0.002
```

Outputs under `profy/results/experiment_YYYYMMDD_HHMMSS/`:
- `models/`: checkpoints (`sensor-_fold*.pth`, `audio-_fold*.pth`)
- `logs/`: training logs
- `figures/`: summary plots

### 3) Full Experiment (Save Checkpoints)

```
./scripts/run.sh --save-checkpoints --lambda-attention-energy 0.002
```

This runs 3‑fold GroupKFold with all 6,476 samples and stores models for alignment analysis.

### 4) Expert‑Alignment Evaluation (Agreement)

Evaluate highlight agreement with expert annotations (Top‑20 set, per annotator and consensus). Use the latest experiment directory and the provided web annotation folder:

```
./scripts/run_expert_alignment.sh \
  results/experiment_YYYYMMDD_HHMMSS \
  ../piano-performance-marker/web_evaluations \
  ../piano-performance-marker/public/audio \
  ../piano-performance-marker/public/audio/manifest_top20.json \
  profy/data \
  --consensus mean --consensus-thr 0.5 --min-evaluators 3
```

Outputs under `results/expert_alignment_YYYYMMDD_HHMMSS/`:
- `summary.json`: per‑annotator aggregate metrics
- `summary_consensus.json`: consensus metrics (vote proportion ≥ threshold)
- `per_audio_metrics.csv`, `per_audio_metrics_consensus.csv`
- `figures/overlay_*`: overlays for individual raters and consensus

## Tips and Notes

- Debug vs Full: debug uses ~60 samples for fast verification; full uses the entire corpus.
- Gating: model logs mean modality weights and entropy; decision‑level fusion (PoE) is used for robust metrics, while mid‑level attention/evidence feeds the UI and alignment evaluation.
- Reproducibility: `config_manifest.json` is saved under each experiment directory; alignment scripts read `results/…/models` and tune only lightweight post‑processing (smooth/lag/power).
