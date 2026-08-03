# Codex Handoff

最終更新: 2026-08-03 (JST)

この文書は次のチャットの開始地点である。現在の区切りは、Zero用Character LoRAの初回学習・評価に加え、長い人物定義下でのprompt優先制御（T-018）と、1〜8枚の逐次生成・画風設定の保存と履歴復元（T-019）を実装してローカル実機確認まで終えた時点。T-018とT-019は完了した。Character LoRAはcheckpoint 720が正面の瞳と猫型髪飾り、600が背面のお団子に優れるが、単一重みで全条件を満たさないためT-017は未完了。

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
2. 人数、Character固定属性、今回の表情・ポーズ、シーン固有タグ、画風・品質の順にタグを統合する
3. Animagine XL 4.0 Zeroで人物と背景を同時に生成する

1回に1〜8枚を指定できる。人物・シーン最適化と生成設定はバッチ内で共通にし、1枚目は指定Seed、2枚目以降は重複しないランダムSeedだけへ変更して、GPUで1枚ずつ直列生成する。

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
- Qwenが一般語へ縮約しても、明示された動作、場所、時間帯を短いタグで最終promptへ再挿入
- 長いpromptはCharacter固定属性、今回の動作、シーン固有タグを画風・品質語より優先してCLIP上限内へ圧縮
- シーン込み全身生成では縦長キャンバスと単一構図を優先し、複数人物・分割構図を抑制
- 背景変更時に元シーンをnegative promptへ追加
- 海への置換やピース指定の補助制約
- 一貫再生成メタデータの保存
- 標準t2iでは `background_mask` を生成しない
- 1〜8枚のSeed列を作り、同じprompt・設定のまま1枚ずつ生成してSSEへ逐次送信

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
- 新規生成と一貫再生成の生成枚数入力、Seed付き複数結果一覧、選択画像の保存・再生成
- 4つの画風強度をブラウザへ保存し、再読み込み後に復元
- 生成履歴へ画風強度を保存し、「プロンプトを履歴から参照」で入力と同時に復元
- シーン込み生成は一回のAPIジョブ。自動inpaintは開始しない
- 手動修正はユーザー指定マスク必須

### LoRA

- `lora_training.py`: 学習ジョブ、メタデータ、状態管理
- `scripts/train_character_lora.py`: SDXL LoRA学習
- `scripts/smoke_lora_adapter.py`: LoRAのロード確認
- `scripts/evaluate_character_lora.py`: 同一Seedでbaselineと複数LoRA重みを比較生成
- ギャラリー画像の右クリック、長押し、メニュー、拡大画面から学習画面を開ける
- 一枚学習は試験用。実用では人物同一性と可変要素を分離した複数画像を使用する
- 画像別caption、縦長アスペクト比bucket、左右反転なしをCharacter LoRAの標準とする
- 学習画像の固定背景は `training_leakage_tags` として保存し、要求されていない場合はnegative promptへ追加する
- 学習画面で日本語の人物固定定義を保存し、LoRA選択時に人物入力の前へ自動挿入する
- CLIP上限では髪、瞳、顔、体格、ポーズを衣装・画風タグより優先する
- `sdxl-animagine-zero`を正確な互換profileとして保存し、旧Opt/JANKU LoRAをZeroへロードしない
- CharacterとStyleを別セレクタで選び、Diffusersの複数adapterとして同時適用する
- Characterはrank 16、`1e-4`、左右反転なし、Styleはrank 32、`5e-5`、左右反転ありを既定とする
- 人物の固定除外条件を保存し、`side bun, bun beside ear`などをnegative promptへ自動挿入する

## 5. Zero用Character LoRA教師画像

### 正本と現在の状態

- ルート:
  - `input/学習用テストデータ/金髪ショートヘアの少女_Zero再学習/`
  - 絶対パス: `C:\Users\GT-1096D\Documents\イラスト生成アプリの開発\input\学習用テストデータ\金髪ショートヘアの少女_Zero再学習`
- 採用36枚:
  - `01_ターンアラウンド`: 12枚
  - `02_表情差分`: 12枚
  - `03_全身ポーズ`: 6枚
  - `04_全身衣装背景`: 3枚
  - `05_全身衣装背景2`: 3枚
- 2026-07-31の機械確認:
  - 採用合計36枚
  - SHA-256完全重複0
  - 最小短辺640px
  - 画像別 `.txt` 36件、欠落0、空caption 0
  - trigger追加後の最大caption長はZeroの両CLIP tokenizerで71トークン、77超過0
- 学習対象外:
  - `00_カタログ/expression_catalog.jpg`: 複数パネルと文字を含むため除外
  - `99_要修正/01_fullbody_front_bun_visible_at_side.jpg`: 正面でお団子が側頭部に突出したため除外
  - `非採用/`: 重複画像など2枚

### 確定した人物仕様

- 金髪の肩付近までの髪
- 後頭部中央の小さな編み込みお団子は1つ
- 正面では後頭部のお団子を無理に見せない
- 横、背面、振り返りでお団子の中央位置と形を教える
- 前髪側の白い猫型髪飾り。視点の反対側で隠れる場合は無理に描かない
- 瞳色は採用画像を基準とするRose Crimson
  - caption・固定人物定義の基準表現: `rose crimson eyes, deep reddish-pink irises`
  - 旧文書の `#ff66cc` 固定は最終仕様ではない
- 小柄で、幼さの残る顔立ち

顔・瞳・髪飾りの主基準は `01_ターンアラウンド/01_front.jpg`、お団子の位置と形の主基準は `01_ターンアラウンド/12_bun_closeup_top_back.jpg`。

### データ構成上の注意

- 正面画像でお団子が見えないことは欠落ではなく、正しい立体配置
- 屋上画像には猫柄のTシャツとバッグ、室内手振り画像には背景の猫がある。これらは人物固有要素ではないため、captionで衣装・背景の可変要素として明示する
- その他の追加シーンでは、猫型髪飾り以外の猫・猫柄を増やしていない
- 白背景の幾何基準だけでなく、教室、屋上、室内、図書館、廊下と複数衣装・ポーズを含む
- `input/` は `.gitignore` 対象。教師画像はGitに含まれず、この文書だけをpushしても別PCへ複製されない

### 学習用ステージングと初回学習

`scripts/train_character_lora.py` の `load_training_records()` は指定datasetの直下だけを `iterdir()` し、サブフォルダを再帰走査しない。現在の正本ルートをそのまま `--dataset` へ渡すと画像0枚になる。

この制約に対して、正本を直接移動・上書きせず、次の別ステージングを作成済み。

- `input/学習用テストデータ/金髪ショートヘアの少女_Zero再学習_学習用フラット/`
- 画像36、同名 `.txt` 36
- 正本とのcaption SHA-256不一致0
- `00_カタログ`、`99_要修正`、`非採用`は含めていない

初回学習:

- 基盤: `cagliostrolab/animagine-xl-4.0-zero`
- trigger: `nkt_chr001`
- rank 16、解像度768、学習率 `1e-4`、左右反転なし、seed 42
- 720ステップ、`--save-every 200`
- dry-run: 画像36、ユニークcaption 35、6種類のbucket
- model id: `a45c8b0d48f44490bd656227730a3f09`
- 保存重み: 200、400、600、720
- 720ステップ版を暫定登録し、smoke load成功

## 6. APIで追加された主な入力

`POST /api/generate/start` の標準生成で次を扱う。

- `character_prompt`
- `scene_prompt`
- `source_scene_prompt`
- `generation_intent`
- `reference_image`
- `reference_strength`
- `batch_count`（t2iのみ1〜8、手動局所修正は常に1）
- `style_settings`
- Character LoRA選択と適用強度
- Style LoRA選択と適用強度

複数生成では各画像を `batch_image` SSEイベントとしてSeed・番号とともに返す。完了メタデータには、生成意図、人物・シーン・統合プロンプト、元シーン、参照利用、参照強度、LoRA情報、画風強度、生成枚数を含める。

## 7. 環境とモデル

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
- Vast.aiの標準prefetchはAnimagine XL 4.0 Zero、SDXL IP-Adapter Plus、Qwen3.5-9Bだけ
- JANKU、Waifu-Inpaint、anime segmentation、FLUX、Qwen-Image-Edit、HiDreamの環境変数は標準テンプレートへ設定しない
- HiDreamソース・追加依存は `HIDREAM_RUNTIME_SETUP_ON_START=1` の明示的なopt-inでのみセットアップする
- Worker経由のギャラリーR2保存ではVast側のCloudflare APIトークンと汎用R2 S3認証は不要
- ユーザー別LoRA永続化では `LORA_R2_*` のbucket-scoped S3認証をVastへ設定し、`ai-model-cache/loras/v1/owners/<owner-key>/models/<model-id>` を正本にする
- Character、Style、将来のPose・Backgroundは同じR2形式で分離し、Vastの `/models/loras` は遅延復元するローカルキャッシュとして扱う
- 学習完了後は推論重みと学習設定を自動保存する。教師画像・captionは `LORA_R2_INCLUDE_TRAINING_DATA=1` の場合だけ保存する

ローカル操作はREADMEと `scripts/local.ps1` を正本にする。既存コンテナが動いている場合、実行状態を確認してから再作成すること。

## 8. 2026-07-28〜31の確認結果

- 既存フロー、Zero・複数LoRA、prompt優先制御、逐次複数生成、画風履歴、ユーザー別R2同期を含む自動テスト64件成功
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
- 7月29日の暫定仕様では瞳を `#ff66cc` 基準として評価した
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
- 7月31日にZero再学習用の修正版36枚を確定
  - ターンアラウンド12、表情12、全身基準6、衣装・背景・ポーズ差分6
  - 採用36枚に完全重複なし
  - 旧 `#ff66cc` 固定を最終仕様から外し、採用画像基準のRose Crimsonへ更新
  - 正面ではお団子を描かず、横・背面・振り返りで後頭部中央の形を教える
  - カタログ、側頭部のお団子画像、重複画像を学習対象外へ分離
- 採用36枚を最終目視監査し、画像別caption 36件を作成
  - 見えている人物固定要素だけを記録し、視点、表情、ポーズ、衣装、背景と分離
  - 猫柄Tシャツ、猫柄バッグ、背景の猫を人物固有の猫型髪飾りと分離
  - trigger追加後もZeroの両CLIP tokenizerで最大71トークン
- Zero用Character LoRAを720ステップまで学習
  - id: `a45c8b0d48f44490bd656227730a3f09`
  - 200: `094bb2353e25076b9d0e434eeec7655f9d2fa0846a68c5fa4f9ba72f47f07c90`
  - 400: `eb23af3bb68629311365d8167208459dcaf63429edd98c089d5cbdab0f7ada25`
  - 600: `0220547c99feddb10054b0c49f2cc1a75b956312cf587ec144ab565889641898`
  - 720: `6a1af2389d1fa89359285b66a7d50456177e2f5be05799474a3fd24f9da78668`
  - 720を暫定重みとして登録し、Zeroパイプラインへのsmoke loadに成功
- 正面、正面のお団子指定なし、斜め、背面、全身でbaselineと4 checkpointを同一Seed比較
  - 正面プロンプトからお団子を外すと側頭部のお団子が消え、720がRose Crimsonの瞳と猫型髪飾りを最もよく再現
  - 背面中央のお団子は600が最良で、720では消失
  - 全身は全checkpointで教師画像より高身長・長脚になる
  - 単一checkpointで完了条件を満たさないため、720を `provisional_not_accepted` としてT-017を継続
- 比較画像、生成条件、所見は `output/t017-zero-lora/` に保存
- ユーザー指定のVサイン、ガラスの林檎のピアス、田んぼ道、放課後、夕方を最終promptへ保持し、実Web生成で単一人物・背景・構図が概ね改善したことをユーザー確認済み
- 実Web UIで2枚をSeed `32` とランダムな別Seedで直列生成し、Seed付き2列結果と選択画像の切替を確認
- 画風強度を73へ変更して再読み込み後も73を維持し、20へ変更後に最新履歴を参照すると73へ戻ることを確認
- 履歴適用後の再読み込みでも73を維持し、ブラウザコンソールエラー0件
- R2模擬環境でCharacter・Styleを同じ利用者だけへ復元し、別利用者へ混入しないことを確認
- 学習済み成果物のメタデータ、推論重み、学習設定を保存し、教師画像・captionが既定では送信されないことを確認
- 教師画像保存を明示的に有効化した場合の保存・復元と、破損重みのSHA-256拒否を確認
- ローカルLoRAを指定したcloud owner keyへ公開し、対応する利用者だけが復元できる移行経路を確認
- 推論に不要な中間checkpointを既定で除外し、再公開時に参照されなくなった同一モデルのR2成果物を整理することを確認
- 変更Pythonファイルの構文、Compose設定、Docker buildが成功し、イメージに秘密情報ファイルとローカル出力が含まれないことを確認

### 実機で残った制約

「背景を海に変更。ポーズを顔元でピースに変更。」では、海、単一人物、ピンク髪は安定したが、Vサインは手を頬へ添えるポーズなどへ変化する場合があった。テキスト制約を増やしても毎回保証できないため、次はOpenPose ControlNetを実装する。

v3は旧全身16、顔18、髪型13、低身長の体格4、画風が近い別衣装・実背景10の合計61枚を使用した。99枚中、高身長化した体格4枚、線と陰影が別系統の背景20枚、冗長な旧全身14枚は除外した。現在のWeb UI上限はCharacter 100枚、Style 150枚へ拡張した。

v3の400ステップ版はv2と600・800ステップ版より小柄な比率を保ちやすく、`pink eyes`と白い猫型髪飾りも再現した。ただしAnimagine Opt用でありZeroにはロードしない。また髪型教師画像の正面・斜め向きに側頭部へ突出したお団子が含まれるため、そのままZeroで再学習しない。

実Web APIで判明した、長い人物固定定義により`seaside train station, ocean, sunset`が一般的な屋外タグへ縮約される問題はT-018で修正した。明示された場所・時間帯はQwen出力とは別に保持して最終promptへ再挿入する。小さなアクセサリーや複雑な手指はpromptへ残っても毎回の視覚再現を保証できず、必要ならLoRA・追加学習・OpenPoseを併用する。

Zero用初回LoRAでは、正面にも固定人物定義の `one small braided bun centered at the back of the head` を入れると側頭部へお団子が出やすい。同じSeedでこの指定だけを外すと誤配置が消えたため、正面から見えない後頭部要素を全視点へ常時挿入する方法は見直す。checkpoint 600と720の長所が分かれ、小柄な全身比率も未達のため、現時点の重みを合格扱いにしない。

## 9. テスト

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

2026-07-31の最終確認では、自動テスト57件、`app.py`・`prompt_refiner.py`・`sdxl_janku_workflow.py` のPython構文、Docker build、`git diff --check` が成功した。実行環境にNode.jsがないため `node --check static/app.js` は未実施だが、実ブラウザで複数生成・履歴復元まで操作し、コンソールエラー0件を確認した。T-017では画像36・caption 36・欠落0・空caption 0、Zeroの両CLIP tokenizerで77トークン超過0、dry-run、720ステップ学習、LoRA smoke load、5条件の実GPU比較も確認済み。2026-08-03のVast現行モデル限定化後も自動テスト57件、主要Python 5ファイルとBashの構文、Compose設定、Docker buildが成功し、ビルド済みイメージに `docs/KEYS.md` と `output/` が含まれないことを確認した。

2026-08-03のT-020確認では、ユーザー別R2同期を含む自動テスト64件、変更Pythonファイルの構文、Compose設定、Docker buildが成功した。ビルド済みイメージに同期コードと移行CLIが含まれ、`docs/KEYS.md` と `output/` が含まれないことも確認した。現在のZero Character LoRAはR2の `local` ownerへ推論重みと学習設定の2成果物、93,065,511 bytesとして保存し、中間checkpoint 4件を整理した。空の一時ストアへ1モデルを復元し、サイズ・SHA-256検証にも成功した。Cloudflare Access利用者ownerとVast.aiを使う復元・生成は未実施。

## 10. 次に行うこと

T-020のネット反映と実R2確認を先に行い、その後にT-017を次の順で続ける。

1. `AGENTS.md`、この文書、`PROJECT_STATUS.md`、`TASKS.md`、`DECISIONS.md`を読む
2. Cloudflare Access経由の `GET /api/lora/models` からowner keyを確認し、現在のZero LoRAをそのownerへ移行する
3. Vast.aiでLoRA一覧への復元と選択生成を確認する
4. `output/t017-zero-lora/EVALUATION_SUMMARY.md` と5つの比較フォルダを確認する
5. 正面では後頭部のお団子を固定人物定義へ常時入れず、背面・斜め向きでは指定する視点依存の扱いを設計する
6. 体格教師画像とお団子教師画像の比率、caption、repeat数を見直す。正本36枚を上書きせず、再学習案ごとにステージングを分ける
7. rank 16、解像度768、学習率 `1e-4`、左右反転なしを基準に再学習し、200ステップごとの比較重みを保存する
8. 正面、左右斜め、背面、全身、別衣装・別背景を複数seedで比較する
9. Rose Crimsonの瞳、顔、後頭部中央のお団子、猫型髪飾り、小柄な全身比率、可変衣装・背景追従を単一checkpointで満たす場合だけCharacter LoRAを採用する
10. Character LoRA合格後に、同じ画風で別人物・別背景・複数構図を含む50枚以上からStyle LoRAを学習する

Cloudflare Access利用者ownerへのLoRA移行とVast.ai復元・生成を先に完了し、その後にCharacter LoRA評価、Style LoRA用50枚以上、T-015のOpenPoseへ進む。T-018とT-019は完了済み、T-020はGitHub・Docker Hub公開と `local` ownerの実R2確認済みで、Access利用者のend-to-end待ち。

## 11. Gitとデプロイ

- ブランチ: `codex/animagine-zero-dual-lora`
- 今回のVast構成整理前のHEAD: `e84a7a6490d7fc74b0e4dd5acc7432ad61f27aad`
- Zero統一、LoRA互換性保護、Character/Style同時適用はローカル開発環境へ実装済み
- Vast.ai本番反映、Cloudflare反映は未実施
- 2026-08-01にローカル評価画像の `output/` をDocker build対象外へ修正し、`c8233dcaba06d4cf180bd31e9bc8006b191faf34` をpushした
- Vast.ai用Docker Hubタグは `nukota0615/hidream-o1-image:c8233dcaba06d4cf180bd31e9bc8006b191faf34`
- 公開digestは `sha256:f9f6df2467bb58725f677005c19a957a61aac98ac79cc2be184a64a1ab3ab44b`、platformは `linux/amd64`
- GHCRはローカル認証切れのため同タグ未公開。Vast.aiではDocker Hubタグを使用する
- Vast.aiは最初のホストでcontainer layer展開に失敗したが、別ホストへ変更後に現行公開イメージの起動と通常イラスト生成に成功した
- 2026-08-03に新しいCloudflareアカウントAPIトークンがactiveであり、Worker、R2、D1の読み取りに成功した
- 新しいR2 S3認証で画像バケットとモデルキャッシュの読み取りに成功した。S3 endpointはCloudflare公式のアカウントID形式を使用する
- 既存WorkerのR2・D1・Access・Backend bindingとD1テーブルを確認したが、デプロイは2026-07-22のままである
- 現在の `BACKEND_SHARED_SECRET` は `plain_text` bindingのため、Vast URL確定後にWorker Secretとして再設定する
- 検証用のCloudflare APIトークンとR2 S3キーは本番稼働前にローテーションする。値はGit・文書へ保存しない
- 2026-08-03にVast.aiテンプレートとCloudflare環境変数の設定が完了し、現在使用中の3モデル系統だけを準備する起動構成へソースと環境変数例を変更した。新しいSHA固定イメージのVast実機起動はユーザーが行う
- 現行モデル限定イメージは `nukota0615/hidream-o1-image:4b7d43c33ce7452cd4a7cb9e6da90011d9d6840f`。リモートdigestは `sha256:14ef1788ac1b8dcc632906e065cd60274248e0114a8a918be8319865ca053df5`、実行platformは `linux/amd64`
- 上記イメージは別のVastホストで起動と通常イラスト生成に成功した。参照生成は未確認
- T-020のユーザー別LoRA R2同期とcheckpoint除外修正は `9b1f096ce4ca5ad4e54871bba7b095f603b41479` としてGitHubへpush済み
- 同コミットのDocker Hubタグは `nukota0615/hidream-o1-image:9b1f096ce4ca5ad4e54871bba7b095f603b41479`、リモートdigestは `sha256:196e8377cf1a5ad0969ec042b74a086a03087ad9759e241203a1980363915f02`、実行platformは `linux/amd64`
- 現在のZero LoRAはR2の `local` ownerへ保存・一時復元済み。Cloudflare Access利用者ownerへの移行とVast.ai生成は未実施
- `input/` はGit管理外。教師画像36枚はローカルにだけ存在する
- `output/t017-zero-lora/` はGitへ未追加。比較画像と評価記録は現在ローカルにだけ存在する
- T-017のcaption・ステージング・比較画像はローカル資産として維持し、T-018とT-019のソース・テスト・管理文書はこのブランチへcommit・pushする

## 12. 関連文書

- `AGENTS.md`: 作業手順
- `docs/PRODUCT_SPEC.md`: 現行仕様
- `docs/PROJECT_STATUS.md`: 現在状態
- `docs/TASKS.md`: 対象タスク
- `docs/DECISIONS.md`: 設計判断
