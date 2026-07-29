# Codex Handoff

最終更新: 2026-07-29 (JST)

## 1. 現在の目的

ギャルゲー・ADV・ライトノベル向けに、次を満たすイラスト生成・編集Webサービスを開発している。

- 背景、表情、ポーズを変えてもキャラクターの一貫性を保つ
- 日本語の人物属性と変更指示への追従性を高くする
- 自動マスクに由来する背景縁、人物透過、髪の欠損をなくす
- Character LoRAと生成メタデータを将来のADV・ライトノベル制作ツールから再利用する

## 2. 採用した生成方式

標準方式は、人物を切り抜いて背景だけをinpaintする二段階方式ではない。人物と背景を一回の全画面生成で描く。

### 新規の一枚絵

1. 日本語の人物定義とシーン指示を別々にQwenで最適化する
2. 人数、今回の表情・ポーズ、人物属性、シーンの順にタグを統合する
3. Animagine XL 4.0 Zeroで人物と背景を同時に生成する

### 一貫再生成

1. 元画像をSDXL IP-Adapter Plusの参照として渡す
2. Character LoRAとStyle LoRAがあれば独立強度で同時に適用する
3. 保存済み人物定義と、新しい背景・表情・ポーズ指示を渡す
4. 元シーンを負の条件へ移し、背景を含む一枚絵として再生成する

IP-Adapter参照は人物中心を切り出し、旧背景の影響を減らす。既定強度は25%。人物維持を強める場合は30〜35%、新しいポーズ・構図を優先する場合は20〜25%を目安にする。

### 手動局所修正

Waifu-Inpaint-XLは、ユーザーがマスクした目、口、手、装飾などの局所修正だけに使う。自動人物抽出、クロマキー、輪郭マスク、背景inpaintは標準UIから実行しない。

## 3. 役割分担

- Character LoRA: 長期的な顔、髪、体格、デザインの同一性
- Style LoRA: 複数キャラクター間で共有する線画、塗り、色設計
- IP-Adapter Plus: 選択した一枚の外見を現在の生成へ即時反映
- 人物テキスト: 固定属性と今回の衣装・表情
- シーンテキスト: 背景、カメラ、照明、今回のポーズ
- OpenPose ControlNet: 厳密な骨格・手の配置。未実装、次の最優先

LoRAはポーズを固定する機能ではない。テキストとIP-Adapterだけでは顔元のVサインなどの手指ポーズは確率的になる。

## 4. 主要な実装

### `app.py`

- 生成意図: `character_asset`、`story_illustration`、`consistent_regeneration`、`manual_edit`
- 人物タグとシーンタグを分離して統合
- `1girl, solo` / `1boy, solo` など単一主体を優先
- 今回のポーズ、人物属性、シーンの順にタグを優先
- 背景変更時に元シーンをnegative promptへ追加
- 海への置換やピース指定の補助制約
- 一貫再生成メタデータの保存
- 標準t2iでは `background_mask` を生成しない

### `sdxl_janku_workflow.py`

- `generate_with_janku(..., reference_image=None)` がIP-Adapter画像を受け取る
- `prepare_character_reference()` がADV向け人物中心クロップを作る
- SDXL IP-Adapter Plus:
  - repository: `h94/IP-Adapter`
  - subfolder: `sdxl_models`
  - weight: `ip-adapter-plus_sdxl_vit-h.safetensors`
  - image encoder: `models/image_encoder`
- `models/image_encoder` の1280次元CLIP encoderを使用する。`sdxl_models/image_encoder` は1664次元で、現行adapter weightと行列サイズが合わない
- 参照なし生成へ戻るとき `unload_ip_adapter()` を呼び、パイプライン状態を解除する
- Character LoRA、Style LoRA、IP-Adapterの強度は独立

### `prompt_refiner.py`

- `character`: 白背景の人物素材
- `story_character`: 人物属性、衣装、表情、ポーズ。背景を含めない
- `story_scene`: シーン、カメラ、照明、今回の表情・ポーズ変更。人物同一性を書き換えない
- 緑クロマ背景の指示は廃止

### `templates/index.html` / `static/app.js`

- 「一貫再生成 / 修正」タブ
- 人物定義とシーン指示の分離
- 参照強度の既定値25%
- 結果画面・ギャラリーから「この画像を参照して再生成」
- 元の人物定義、シーン、LoRA、生成設定を可能な限り引き継ぐ
- シーン込み生成は一回のAPIジョブ。自動inpaintは開始しない
- 手動修正はユーザー指定マスク必須

### LoRA

- `lora_training.py`: 学習ジョブ、メタデータ、状態管理
- `scripts/train_character_lora.py`: SDXL LoRA学習
- `scripts/smoke_lora_adapter.py`: LoRAのロード確認
- `scripts/evaluate_character_lora.py`: 同一Seedでbaselineと複数LoRA重みを比較生成
- ギャラリー画像の右クリック、長押し、メニュー、拡大画面から学習画面を開ける
- 一枚学習は試験用。実用には同一人物10〜20枚を推奨
- 画像別caption、縦長アスペクト比bucket、左右反転なしをCharacter LoRAの標準とする
- 学習画像の固定背景は `training_leakage_tags` として保存し、要求されていない場合はnegative promptへ追加する
- 学習画面で日本語の人物固定定義を保存し、LoRA選択時に人物入力の前へ自動挿入する
- CLIP上限では髪、瞳、顔、体格、ポーズを衣装・画風タグより優先する
- `sdxl-animagine-zero`を正確な互換profileとして保存し、旧Opt/JANKU LoRAをZeroへロードしない
- CharacterとStyleを別セレクタで選び、Diffusersの複数adapterとして同時適用する
- Characterはrank 16、`1e-4`、左右反転なし、Styleはrank 32、`5e-5`、左右反転ありを既定とする
- 人物の固定除外条件を保存し、`side bun, bun beside ear`などをnegative promptへ自動挿入する

## 5. APIで追加された主な入力

`POST /api/generate/start` の標準生成で次を扱う。

- `character_prompt`
- `scene_prompt`
- `source_scene_prompt`
- `generation_intent`
- `reference_image`
- `reference_strength`
- Character LoRA選択と適用強度
- Style LoRA選択と適用強度

完了メタデータには、生成意図、人物・シーン・統合プロンプト、元シーン、参照利用、参照強度、LoRA情報を含める。

## 6. 環境とモデル

- ローカルURL: `http://127.0.0.1:7861`
- ローカル・Vast共通基盤: `cagliostrolab/animagine-xl-4.0-zero`
- LoRA互換profile: `sdxl-animagine-zero`
- ローカルCompose: `docker-compose.local.yml`
- モデルキャッシュ: `/models/huggingface`
- ローカルPython/CUDA依存: 外部volume `janku-python-local` の `/venv/main`
- IP-Adapterの主な環境変数:
  - `IP_ADAPTER_ENABLED=1`
  - `IP_ADAPTER_REPO_ID=h94/IP-Adapter`
  - `IP_ADAPTER_SUBFOLDER=sdxl_models`
  - `IP_ADAPTER_WEIGHT_NAME=ip-adapter-plus_sdxl_vit-h.safetensors`
  - `IP_ADAPTER_IMAGE_ENCODER_FOLDER=models/image_encoder`
  - `IP_ADAPTER_DEFAULT_SCALE=0.25`
  - `IP_ADAPTER_CHARACTER_CROP=1`
- Waifu-Inpaintとanime segmentationの起動時prefetchは既定で無効
- モデルはDockerイメージへ含めず、永続volumeへ保存する
- 任意HiDream取得が失敗しても、標準生成とZero取得は継続する

ローカル操作はREADMEと `scripts/local.ps1` を正本にする。既存コンテナが動いている場合、実行状態を確認してから再作成すること。

## 7. 2026-07-28〜29の確認結果

- 既存フローとZero・複数LoRAを含む自動テスト46件成功
- Python、JavaScript、Bashの構文確認成功
- Docker Compose設定確認とDockerイメージビルド成功
- ローカルUIでZero表示、Character・Styleの別セレクタ、Style学習時の人物固定項目非表示を確認
- Zero単体で768×1152、seed 424242の海辺・金髪・ピンク色の瞳・白いサマードレスを実生成し、結果profileが `sdxl-animagine-zero` であることを確認
- コンテナ再作成後に外部volumeのPython/CUDA依存が再利用されることを確認
- 実GPUで新規の一枚絵を生成
  - 人物: ピンク色のショートヘアの女子学生、笑顔、全身
  - シーン: 放課後の教室、夕日
- 同画像を参照して海・砂浜へ全画面再生成
  - 単一人物、ピンク髪、制服を維持
  - 教室・窓を除去し、海・砂浜へ変更
  - 自動マスクを使わないためクロマ縁・輪郭マスク残りなし
- 参照あり生成の後に参照なし生成へ戻してもIP-Adapter状態が残らない
- 同一人物30枚から `猫Tシャツの少女 v2` を400ステップで学習
  - id: `2034b54049374990aff1cd619a680983`
  - trigger: `nkt_chr001`
  - model: `sdxl-animagine`
  - 推奨LoRA強度: 0.8
  - 推奨参照強度: 人物優先0.25、ポーズ優先0.20
- LoRAなし、LoRAのみ、LoRA＋参照25%を同一Seedで比較
- ユーザー確認済み仕様とデータ監査結果を、`#ff66cc`基準のピンク色の瞳、後頭部中央のお団子、前髪側の白い猫型髪飾り、小柄で幼い顔立ちへ訂正
- 固定人物定義を省略したAPI入力でも、LoRA選択時に上記定義が最終プロンプトへ自動挿入されることを実GPU確認
- 白背景由来の白枠をモデル別negative制約で解消
- 追加99枚を顔、全身、体格、髪型、要素分離の観点で検査し、体格と画風が一貫した61枚をv3へ採用
- `金髪ショートヘアの少女 v3（選別61枚）` をrank 16、768解像度、左右反転なしで800ステップまで学習
  - id: `3859464ec2be41cba66a3ad500b9b7de`
  - trigger: `nkt_chr001`
  - 200・400・600・800ステップを保存
  - 体格と髪飾り位置の比較から400ステップ版を採用
  - 推奨LoRA強度: 0.8
- 認証ありローカル環境でもsmokeスクリプトが`BACKEND_SHARED_SECRET`を`X-Backend-Key`として送るよう修正

### 実機で残った制約

「背景を海に変更。ポーズを顔元でピースに変更。」では、海、単一人物、ピンク髪は安定したが、Vサインは手を頬へ添えるポーズなどへ変化する場合があった。テキスト制約を増やしても毎回保証できないため、次はOpenPose ControlNetを実装する。

v3は旧全身16、顔18、髪型13、低身長の体格4、画風が近い別衣装・実背景10の合計61枚を使用した。99枚中、高身長化した体格4枚、線と陰影が別系統の背景20枚、冗長な旧全身14枚は除外した。現在のWeb UI上限はCharacter 100枚、Style 150枚へ拡張した。

v3の400ステップ版はv2と600・800ステップ版より小柄な比率を保ちやすく、`pink eyes`と白い猫型髪飾りも再現した。ただしAnimagine Opt用でありZeroにはロードしない。また髪型教師画像の正面・斜め向きに側頭部へ突出したお団子が含まれるため、そのままZeroで再学習しない。

実Web APIのv3評価では、`seaside train station, ocean, sunset`が長い人物固定定義との結合後に`outdoors, blue sky`へ縮約された。人物属性を守りながらシーン固有タグを残す修正をT-018として記録した。

## 8. テスト

代表コマンド:

```powershell
docker exec -w /workspace/janku-image-studio janku-image-studio-local /venv/main/bin/python3 -m unittest discover -s tests -p "test_*.py"
docker exec -w /workspace/janku-image-studio janku-image-studio-local /venv/main/bin/python3 -m py_compile app.py prompt_refiner.py sdxl_janku_workflow.py image_edit_workflows.py lora_training.py
docker exec -w /workspace/janku-image-studio janku-image-studio-local node --check static/app.js
docker exec -w /workspace/janku-image-studio janku-image-studio-local bash -n deploy/vast/entrypoint.sh
docker compose -f docker-compose.local.yml config
docker compose -f docker-compose.local.yml build
git diff --check
```

Node.jsが実行コンテナにない場合は、Nodeを含む既存ビルド環境で `node --check static/app.js` を実行する。

## 9. 次に行うこと

1. T-017: 後頭部中央のお団子が全視点で一貫した修正版データでZero用Character LoRAを再学習する
2. T-017: 同じ画風で別人物・別背景・複数構図を含む50枚以上からStyle LoRAを学習する
3. T-018: 長い人物固定定義と同時に海・駅・夕方などのシーン固有タグを保持する
4. T-015: SDXL OpenPose ControlNetとポーズ参照入力を実装する
5. 最新コードをVast.aiで起動し、参照生成と通常生成を確認する
6. Worker・Vast.ai・R2・D1のend-to-endを再検証する

## 10. Gitとデプロイ

- ブランチ: `codex/animagine-zero-dual-lora`
- Zero統一、LoRA互換性保護、Character/Style同時適用はローカル開発環境へ実装済み
- Vast.ai本番反映、Cloudflare反映は未実施

## 11. 関連文書

- `AGENTS.md`: 作業手順
- `docs/PRODUCT_SPEC.md`: 現行仕様
- `docs/PROJECT_STATUS.md`: 現在状態
- `docs/TASKS.md`: 対象タスク
- `docs/DECISIONS.md`: 設計判断
