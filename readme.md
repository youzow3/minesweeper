# 使い方

依存関係のインストール

```bash
pip install -r requirements.txt
```

もしくは

```bash
pip install icecream
```

agent.pyを使う場合は前者のコマンドを使ってください

## minesweeper.py

実行

```bash
python3 minesweeper.py
```

### パラメーター

* size

    盤面のサイズ、カンマ区切り、デフォルト: 10,10

* bomb

    爆弾の数、デフォルト: 10

* position

    スタート時に安全な場所、カンマ区切り、デフォルト: なし

* debug

    デバッグモード、デフォルト: 無効

* color

    色付きの出力を行う(環境依存)、デフォルト: 無効

### 例

8x8,爆弾16個で開始、カラー出力

```bash
python3 minesweeper.py --size 8,8 -bomb 16 --color
```

### 遊び方

毎回盤面を表示したあと、コマンド待機状態になります。

#### コマンド

* h

    ヘルプを表示

* o [x] [y]

    指定した座標のマスを開く。もし爆弾なら爆発してそのまま終了します。

* f [x] [y]

    フラグを建てる。

* c

    ゲームをクリアできているかチェックします。クリアしていたらそのまま終了します。

## agent.py

AIに学習してもらおうとして作った残骸です。残してあるだけです。
