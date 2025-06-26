# Streamlit Cloud Secrets設定ガイド

## 🚨 重要: OpenAI API Key必須設定

このアプリはOpenAI APIを使用したOCR校正機能が必須です。
APIキーが未設定の場合、アプリは起動しません。

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

- ✅ **必須**: Streamlit Cloud Secretsに設定
- ❌ **削除済み**: アプリ内での手動入力機能
- ❌ **禁止**: GitHubリポジトリにAPIキーをコミット

## API接続確認

アプリ起動時に自動的にAPI接続テストが実行されます:
- ✅ 接続成功: サイドバーに「OpenAI API接続確認済み」表示
- ❌ 接続失敗: エラーメッセージ表示でアプリ停止

## トラブルシューティング

### APIキーが認識されない場合
1. Secretsの形式を確認（TOMLフォーマット）
2. アプリを手動で再起動
3. 「Manage app」→「Reboot」を実行

### API接続エラーの場合
1. **無効なAPIキー**: キーの形式・スペルを確認
2. **残高不足**: OpenAIアカウントの課金設定確認
3. **レート制限**: 少し時間をおいて再試行
4. **ネットワークエラー**: Streamlit Cloudの一時的な問題

### APIキーの形式
- OpenAI APIキーは `sk-proj-` または `sk-` で始まる
- 英数字とハイフンのみ含む
- 通常50文字以上の長さ

### エラーメッセージ例
- `Invalid API key`: APIキーが間違っている
- `Insufficient quota`: 使用量制限に達している
- `Rate limit exceeded`: リクエスト制限に達している