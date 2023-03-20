# FFVAE: Flexibly Fair Representation Learning by Disentanglement
非公式のFFVAE (https://arxiv.org/pdf/1906.02589.pdf) の再現実装です．実装にあたって，[disentangling-vae](https://github.com/YannDubs/disentangling-vae/tree/f0452191bab6d94eba0b4e6a065f74dcfd54ac52)を参考にしています．

## 実行環境
docker-composeを用います．インストールしていない人は公式のドキュメントに従ってインストールしてください．

- Dockerイメージの作成
```bash
$ make build
```
- Dockerコンテナ起動
```bash
$ make up
```
- Dockerコンテナに入る
```bash
$ make exec
```
## 実行方法
```bash
$ python main.py -c config/celeba.yaml
```
