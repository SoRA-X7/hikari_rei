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
  --seed 1 --output runs/ultra/hikari-rust/replay.json

# JSON行単位のTetris Bot Protocol。startから継続して先読み
target/release/hikari_rei tbp --threads 1
```

依存クレートのキャッシュがない環境では、初回だけ `--offline` を外してください。

`--threads` は展開・評価の並列ワーカー数、`--batch-size` は一度に選ぶ葉の数です。既定値はそれぞれ 1 と 128 です。`--max-nodes` の既定値は 2,000,000。上限に達すると既存の探索結果を保持して展開を止め、手の確定によって先頭層が解放されると再開します。

`bench --position position.json` は `{"board": [0, 0], "queue": ["T", "I", "O"]}` のような入力を受け付けます。盤面は下の行から順に並べた10ビットの行マスク、またはTBPのセル配列です。

`tbp` は `rules`、`start`、`suggest`、`play`、`new_piece`、`stop`、`quit` に対応します。カウントダウン開始前に確定盤面と公開nextを `start` に渡せます。`suggest` は公開済みの探索結果を読み、`play` で実際に置いた手を確定するまでルートを進めません。`stop` は探索停止の完了を待ちます。拡張メッセージ `reset` に盤面を渡すと、queueとbagを保って盤面を再同期します。純粋なお邪魔の追加なら、旧計画をせり上がり分ずらして探索を誘導します。

`ultra` はHikari規則による180秒のプレイ記録モードです。3秒のカウントダウン前から探索し、移動・ソフトドロップ・待ち時間の間も探索を継続します。公開情報は現在のミノと5個のNEXTで、手の確定後に盤面とqueueの一致を確認します。攻撃は `SimpleBoard` の値を集計し、スコアは未実装です。

グラフ登録には、盤面全体を比較する専用の照合表を既定で使います。参照用の登録方式で動かす場合は `--no-default-features` を指定してください。

`fast-low-stack` featureは、低い盤面で空中の移動・回転経路を事前計算します。`prune-low-stack` はさらに地形の上へ戻る状態を省略する近似探索です。配置の取りこぼしや操作コスト・順序の変更があり得るため、両方とも既定では無効です。

`profile-search` featureは、配置生成・評価・グラフ登録などの工程別実時間とCPU時間を計測します。`examples/profile_probe.rs` と `tools/profile_search.py` で計測できます。通常ビルドでは計測用のタイマーと集計フィールドは除かれます。

生成したリプレイ・ベンチマーク結果・ログは、Git管理外の `runs/` に保存します。リポジトリの文書には恒久的な仕様・設計・使い方を記載します。
