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

`--threads` は探索処理の並列ワーカー数で、省略するとOSが返す利用可能な並列数を使います。明示した値で上書きできます。通常の探索にはバッチ件数の指定はありません。Rust APIでは `Config::for_threads(threads)` でワーカー数を設定でき、`Config::default()` は1ワーカーです。

探索はフレームレートや固定件数に区切らず連続実行します。コントローラがグラフを管理し、ワーカーが配置生成と評価を行います。完了した仕事を順次登録・逆伝播し、全ワーカーの完了を待たずに次の選択へ進みます。逆伝播は深い層から順に進め、値や生死が変わらない祖先では打ち切ります。同じ層内の十分大きな更新は並列計算し、小さな更新は逐次処理します。Rust APIの `Search::search_step` は完了した仕事を反映し、`Search::pause` は進行中の仕事を停止します。

仕事の投入は最大16葉をまとめ、ワーカー上のFIFOキューへ載せてwork stealingで分担します。共有ハンドルの複製と外部キューへの投入をまとめることで、受け渡しの負担を減らします。端数もすぐに投入し、結果は葉ごとに返します。未完了の仕事はワーカー数の2倍までです。

`--max-nodes` の既定値は 2,000,000。上限に達すると既存の探索結果を保持して展開を止め、手の確定によって先頭層が解放されると再開します。

`bench --position position.json` は `{"board": [0, 0], "queue": ["T", "I", "O"]}` のような入力を受け付けます。盤面は下の行から順に並べた10ビットの行マスク、またはTBPのセル配列です。

`bench --reference-batch-size 128 --ms 3000` は比較用の同期バッチ処理を時間指定で計測します。このオプションは `bench` 専用で、省略すると通常の連続探索です。出力の `search_mode` で使用した方式を確認できます。

`tbp` は `rules`、`start`、`suggest`、`play`、`new_piece`、`stop`、`quit` に対応します。カウントダウン開始前に確定盤面と公開nextを `start` に渡せます。`suggest` は現在の探索結果を読み、`play` で実際に置いた手を確定するまでルートを進めません。応答用JSONは要求時に生成し、探索の進行ごとには生成しません。`stop` は探索停止の完了を待ちます。拡張メッセージ `reset` に盤面を渡すと、queueとbagを保って盤面を再同期します。純粋なお邪魔の追加なら、旧計画をせり上がり分ずらして探索を誘導します。

`ultra` はHikari規則による180秒のプレイ記録モードです。3秒のカウントダウン前から探索し、移動・ソフトドロップ・待ち時間の間も探索を継続します。公開情報は現在のミノと5個のNEXTで、手の確定後に盤面とqueueの一致を確認します。攻撃は `SimpleBoard` の値を集計し、スコアは未実装です。

グラフ登録には、盤面全体を比較する専用の照合表を既定で使います。参照用の登録方式で動かす場合は `--no-default-features` を指定してください。

`fast-low-stack` featureは、低い盤面で空中の移動・回転経路を事前計算します。`prune-low-stack` はさらに地形の上へ戻る状態を省略する近似探索です。配置の取りこぼしや操作コスト・順序の変更があり得るため、両方とも既定では無効です。

`profile-search` featureは、配置生成・評価・グラフ登録などの工程別実時間とCPU時間を計測します。`examples/profile_probe.rs` と `tools/profile_search.py` で計測できます。通常ビルドでは計測用のタイマーと集計フィールドは除かれます。

`profile_probe --ms 3000` は通常の連続探索を計測します。従来の同期バッチ処理は参照・比較用として残し、`--batches 128` で固定回数、`--reference-batches --ms 3000` で時間を指定して計測できます。参照バッチの選択件数は128で、`--batch-size` による変更はこの2つの参照モードでのみ受け付けます。Rust APIの `Config::batch_size` と `Search::search_batch` も参照処理用です。

比較用スクリプトは同じ仕事量を比べるとき参照モードを明示します。連続探索と参照バッチでは選択時に反映済みのグラフが異なるため、同じ時間での探索結果の一致は要求しません。並列逆伝播のCPU時間はコントローラスレッド分のみで、並列ワーカーを含む時間比較には `backpropagation_wall_ns` を使ってください。

連続探索ではワーカーの計算と登録処理が重なります。`expansion_wall_ns` は並列時には完了結果を待った時間で、配置生成そのものの時間は `expansion_worker_*`、候補評価は `evaluation_worker_*` に記録します。`tools/profile_search.py` は通常の連続探索を計測し、`--reference-batches` を付けると参照方式を計測します。

`selection_wall_ns` の内訳として、探索経路を選ぶ時間を `selection_walk_wall_ns`、キュー投入時間を `dispatch_wall_ns` に記録します。`dispatch_groups` は投入回数、`dispatch_jobs` は投入した葉の数です。計測用の `profile_probe` は、`--threads` を省略すると1ワーカーで動きます。

`--batch-size` に未対応の旧 `profile_probe` は128で比較でき、出力された実際の処理量も確認します。それ以外の件数を比較する場合は、両方のprobeを同じCLIでビルドしてください。

`tools/benchmark_registration.py` は、変更前後の通常ビルドと `profile-search` ビルドを比較し、同一仕事量での結果一致、ノード上限、探索速度、登録時間を検証します。既定の照合表同士を比較するときは `--baseline-fast-registration` を指定します。工程別計測は乱数を変えて3回繰り返し、`--profile-trials` で回数を変更できます。別のソースツリーをビルドする場合は、それぞれ異なる `--target-dir` を使ってください。

生成したリプレイ・ベンチマーク結果・ログは、Git管理外の `runs/` に保存します。リポジトリの文書には恒久的な仕様・設計・使い方を記載します。

`tools/versus.py` は2つのTBP実行ファイルを実時間で対戦させます。`cargo build --release --example arena_rules` で共通の判定プロセスを用意し、`--hikari`・`--cold-clear`・`--rules` に各実行ファイル、`--output` に新しい保存先を指定します。既定は両者1ワーカー、6個の公開ミノ、3秒カウントダウン、60Hz・最大180秒を4試合です。2つの開発用シードを使い、同じシードで左右を交代します。

対戦にはHikariの24行盤面・操作コスト・攻撃量を共通で使います。出現待ちは3フレーム、ライン消去待ちは38フレーム。お邪魔は60フレーム後に到着し、攻撃で先着順に相殺、消去しなかった着地で最大8行上昇します。上昇時は両エンジンへ `stop`・`start` を送り、盤面・hold・combo・B2B・公開queue・bagを再同期します。各配置は共通の配置生成で到達確認し、同じ4セルを占める非Tミノの回転は同値として扱います。通信・到達判定の不一致は対戦エラーとして停止し、勝敗に数えません。着地ログ、TBP通信、バイナリのハッシュを保存します。予約シードを使う正式評価とは別の開発用対戦です。
