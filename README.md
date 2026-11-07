# Hikari Rust

[SoRA_X7 の Hikari](https://gitlab.com/SoRA_X7/hikari) を Rust に再実装しています。
移植元は `90aef085a53d6048cbd19de7f4c9c53a6f3c8960` です。

24行の `SimpleBoard`、SRS、Mirai の配置生成と操作手順、整数評価関数、層ごとの盤面共有を行う探索を移植しました。確定した手で先頭層を取り除き、深い層のノード・評価・親子関係を保持します。未知の next は bag から推測し、公開された時点で該当する分岐へ絞ります。

```sh
cargo test --offline
cargo build --release --offline

# 空盤面を3秒探索。ノード数・生成辺数・深さ・候補手をJSONで表示
target/release/hikari_rei bench --ms 3000 --threads 1

# 操作手順と盤面を含む自己対局リプレイを保存
target/release/hikari_rei selfplay --pieces 100 --ms-per-move 20 \
  --countdown-ms 3000 --output runs/hikari-rust/selfplay.json

# 実時間で3分プレイし、入力・盤面・探索統計を記録
target/release/hikari_rei ultra --threads 4 --max-nodes 8000000 \
  --seed 10000000 --output runs/ultra/hikari-rust/replay.json

# JSON行単位のTetris Bot Protocol。startから継続して先読み
target/release/hikari_rei tbp --threads 1
```

依存クレートのキャッシュがない環境では、初回だけ `--offline` を外してください。

`--threads` は展開・評価の並列ワーカー数、`--batch-size` は一度に選ぶ葉の数です。既定値はそれぞれ 1 と 128 です。`--max-nodes` の既定値は 2,000,000。上限に達すると既存の探索結果を保持して展開を止め、手の確定によって先頭層が解放されると再開します。

`bench --position position.json` は `{"board": [0, 0], "queue": ["T", "I", "O"]}` のような入力を受け付けます。盤面は下の行から順に並べた10ビットの行マスク、またはTBPのセル配列です。

`tbp` は `rules`、`start`、`suggest`、`play`、`new_piece`、`stop`、`quit` に対応します。カウントダウン開始前に確定盤面と公開nextを `start` に渡せます。`suggest` は公開済みの探索結果を読み、`play` で実際に置いた手を確定するまでルートを進めません。`stop` は探索停止の完了を待ちます。拡張メッセージ `reset` に盤面を渡すと、queueとbagを保って盤面を再同期します。純粋なお邪魔の追加なら、旧計画をせり上がり分ずらして探索を誘導します。

`ultra` はHikari規則による180秒のプレイ記録モードです。3秒のカウントダウン前から探索し、移動・ソフトドロップ・待ち時間の間も探索を継続します。公開情報は現在のミノと5個のNEXTで、手の確定後に盤面とqueueの一致を確認します。攻撃は `SimpleBoard` の値を集計し、スコアは未実装です。

元コードとの対応、差分、測定結果は [docs/PORT.md](docs/PORT.md) に記録します。自己対局とUltraは元Hikariの盤面規則を使った動作確認です。cold-clear-2との対戦は、今回の移植でまだ測定していません。

低盤面の移動生成を省略する実験機能は既定では無効です。速度と配置の取りこぼしを比較した結果、ビルド方法は [docs/LOW_STACK_EXPERIMENT.md](docs/LOW_STACK_EXPERIMENT.md) にあります。

配置生成・評価・グラフ登録などの処理時間の実測は [docs/SEARCH_PROFILE.md](docs/SEARCH_PROFILE.md) にあります。計測は `profile-search` featureを有効にしたときだけ行います。

置き換え前のNN/MCTS実装は、退避コピー・モデル・実験出力・専用Python環境・加工済み学習データを含めて削除しました。Hikari原典、取得元データ、評価用seed、現在のRust版の検証結果とUltraリプレイは保存しています。
