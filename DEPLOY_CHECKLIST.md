# 🚀 Streamlit Cloud 部署檢查清單

## ✅ 部署前檢查

### 必要檔案
- [x] `ragic_inventory.py` - 主程式檔
- [x] `requirements.txt` - 依賴套件清單
- [x] `.streamlit/config.toml` - Streamlit 設定檔
- [x] `.gitignore` - Git 忽略規則
- [x] `README.md` - 說明文件

### 安全性檢查
- [x] API Key 已改為從環境變數/Secrets 讀取
- [x] `.gitignore` 已排除 `secrets.toml` 和資料庫檔案
- [x] 程式碼中沒有硬編碼的敏感資訊

### GitHub 準備
- [ ] 建立 GitHub repository
- [ ] 上傳所有必要檔案到 repository
- [ ] 確認 `.gitignore` 已生效（資料庫檔案不會被上傳）

### Streamlit Cloud 設定
- [ ] 登入 Streamlit Cloud
- [ ] 連接 GitHub repository
- [ ] 設定 Main file path: `ragic_inventory.py`
- [ ] 在 Secrets 中設定 API Key（見下方說明）

## 🔐 設定 Streamlit Secrets

在 Streamlit Cloud 的應用程式設定中：

1. 點擊 "Secrets" 或 "⚙️ Settings" → "Secrets"
2. 在編輯器中輸入：

```toml
[ragic]
api_key = "MEwyTEExWHJQamRDalZ6N0hzQ2syZlBHNUNJeWhwZFBrM3BMM2tDRWd4aGIvZ1JxWTlYaGkyM0RoRmo1ZExHaA=="
```

3. 點擊 "Save"

## 📝 部署步驟

1. **準備 GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Ragic 秒數控管系統"
   git remote add origin <你的 GitHub repository URL>
   git push -u origin main
   ```

2. **在 Streamlit Cloud 部署**
   - 前往 https://streamlit.io/cloud
   - 點擊 "New app"
   - 選擇你的 repository
   - 設定 Main file path: `ragic_inventory.py`
   - 點擊 "Deploy"

3. **設定 Secrets**
   - 在應用程式頁面點擊 "⚙️ Settings"
   - 選擇 "Secrets"
   - 貼上上述 secrets 內容
   - 儲存

4. **測試**
   - 等待部署完成
   - 測試「強制從 Ragic 更新」功能
   - 確認所有功能正常運作

## ⚠️ 注意事項

- Streamlit Cloud 的檔案系統是**臨時的**，每次重新部署可能會清除資料庫
- 建議定期從 Ragic 更新資料，而不是依賴本地快取
- 如果資料庫很重要，可以考慮使用外部資料庫服務（如 Supabase、PlanetScale 等）

## 🐛 問題排除

### 問題：應用程式無法啟動
- 檢查 `requirements.txt` 是否包含所有必要套件
- 查看 Streamlit Cloud 的日誌輸出

### 問題：無法連線 Ragic API
- 確認 Secrets 中的 API Key 是否正確
- 檢查 API Key 是否有權限存取該 Ragic 表單

### 問題：資料庫無法寫入
- Streamlit Cloud 支援 SQLite，但資料是臨時的
- 每次重新部署會清除資料，這是正常行為
