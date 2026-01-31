関連ドキュメントのURLをプロジェクト内と公式ソースから集めています。

[3 tools called]

Core ML 変換まわりで参照しているドキュメントの URL を整理しました。

---

## Core ML Tools（公式）

| 内容                          | URL                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| ドキュメントトップ            | https://apple.github.io/coremltools/docs-guides/index.html                         |
| インストール（macOS / Linux） | https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html |
| 統一変換 API 概要             | https://apple.github.io/coremltools/docs-guides/source/unified-conversion-api.html |
| PyTorch 変換（ct.convert）    | https://apple.github.io/coremltools/source/coremltools.converters.convert.html     |
| GitHub リポジトリ             | https://github.com/apple/coremltools                                               |
| リリース一覧                  | https://github.com/apple/coremltools/releases                                      |

---

## VibeVoice（モデル・変換関連）

| 内容                                  | URL                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------- |
| Hugging Face モデル（VibeVoice-1.5B） | https://huggingface.co/microsoft/VibeVoice-1.5B                        |
| VibeVoice 公式リポジトリ              | https://github.com/microsoft/VibeVoice                                 |
| TTS 利用ドキュメント（無効化説明）    | https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md |

---

## プロジェクト内ドキュメント（参照用）

| 内容                                         | プロジェクト内パス                                      |
| -------------------------------------------- | ------------------------------------------------------- |
| VibeVoice Core ML 変換ガイド                 | `Docs/VibeVoice-1.5B-CoreML-Conversion.md`            |
| Core ML Tools インストール（ローカルコピー） | `Docs/Core ML Tools/source/installing-coremltools.md` |
| Core ML Tools ガイド概要                     | `Docs/Core ML Tools/CoreMLTools-Guide-Summary.md`     |

---

## その他（依存・参考）

| 内容                | URL                                         |
| ------------------- | ------------------------------------------- |
| Python 公式         | https://www.python.org/downloads/           |
| pip                 | https://pip.pypa.io/en/stable/              |
| Conda               | https://docs.conda.io/en/latest/index.html  |
| venv（Python 標準） | https://docs.python.org/3/library/venv.html |

Windows で変換する場合は、Core ML Tools の「インストール」と「統一変換 API」の公式ドキュメントに加え、WSL2 上で使う場合は Linux 向けの手順を参照する形になります。
