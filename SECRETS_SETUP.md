# Streamlit Cloud Secrets設定ガイド

## OpenAI API Key設定方法

### 1. Streamlit Cloudでアプリを開く
1. [Streamlit Cloud](https://share.streamlit.io/)にアクセス
2. 対象のアプリ（OCRPassages）を選択

### 2. Secrets設定
1. アプリページで「⚙️ Settings」をクリック
2. 「Secrets」タブを選択
3. 以下の形式でAPIキーを追加:

```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 3. 設定例
```toml
[secrets]
OPENAI_API_KEY = "sk-proj-abcdefg..."
```

### 4. 設定後
1. 「Save」をクリック
2. アプリが自動的に再起動
3. サイドバーに「✅ API Key (Secrets)設定済み」と表示されることを確認

## セキュリティ注意事項

- ✅ **推奨**: Streamlit Cloud Secretsに設定
- ❌ **非推奨**: アプリ内での手動入力（毎回入力が必要）
- ❌ **禁止**: GitHubリポジトリにAPIキーをコミット

## トラブルシューティング

### APIキーが認識されない場合
1. Secretsの形式を確認（TOMLフォーマット）
2. アプリを手動で再起動
3. 「Manage app」→「Reboot」を実行

### APIキーの形式
- OpenAI APIキーは `sk-proj-` または `sk-` で始まる
- 英数字とハイフンのみ含む
- 通常50文字以上の長さ