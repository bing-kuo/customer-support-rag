# 客服 RAG

以合成資料驗證客服檢索增強生成（RAG）的 proof of concept。此 repository 目前處於早期開發階段，尚未核准用於正式環境。

## 開發環境

- Python 3.12 或更新版本
- 開發工具版本固定於 `pyproject.toml`

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 本機檢查

```sh
ruff format --check .
ruff check .
pytest
```

GitHub Actions 會執行相同的格式、lint 與測試檢查，並以 Gitleaks 掃描 secrets。

目前只包含 Python 專案骨架與最小 CI。API、RAG 行為、模型整合及合成語料會在後續 PR 分別提出 review。
