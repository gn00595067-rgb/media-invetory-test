# Streamlit Cloud 部署指南

## 📋 部署前準備

### 1. 建立 GitHub Repository

1. 在 GitHub 建立新的 repository
2. 將專案檔案上傳到 repository

### 2. 設定 Streamlit Secrets（重要！）

在 Streamlit Cloud 部署時，需要設定環境變數來保護 API Key：

#### 方法一：使用 Streamlit Secrets（推薦）

1. 在 Streamlit Cloud 的應用程式設定中，找到 "Secrets" 選項
2. 點擊 "Edit secrets"
3. 在編輯器中輸入以下內容：

```toml
[ragic]
api_key = "MEwyTEExWHJQamRDalZ6N0hzQ2syZlBHNUNJeWhwZFBrM3BMM2tDRWd4aGIvZ1JxWTlYaGkyM0RoRmo1ZExHaA=="
```

#### 方法二：使用環境變數

在 Streamlit Cloud 的應用程式設定中，找到 "Environment variables" 選項，新增：
- 變數名稱：`RAGIC_API_KEY`
- 變數值：你的 Ragic API Key

### 3. 部署步驟

1. 登入 [Streamlit Cloud](https://streamlit.io/cloud)
2. 點擊 "New app"
3. 選擇你的 GitHub repository
4. 設定：
   - **Main file path**: `ragic_inventory.py`
   - **Branch**: `main` (或你的主要分支)
5. 點擊 "Deploy"

### 4. 部署後檢查

- ✅ 確認應用程式可以正常載入
- ✅ 測試「強制從 Ragic 更新」功能
- ✅ 確認資料庫可以正常建立和讀寫

## 🔒 安全性注意事項

- ⚠️ **絕對不要**將 API Key 直接寫在程式碼中
- ⚠️ **絕對不要**將 `secrets.toml` 檔案上傳到 Git
- ✅ 使用 Streamlit Secrets 或環境變數來管理敏感資訊
- ✅ `.gitignore` 已經設定好，會自動排除敏感檔案

## 📁 需要上傳的檔案

上傳到 GitHub 的檔案應該包括：
- ✅ `ragic_inventory.py` - 主程式
- ✅ `requirements.txt` - 依賴套件
- ✅ `README.md` - 說明文件
- ✅ `.streamlit/config.toml` - Streamlit 設定
- ✅ `.gitignore` - Git 忽略規則
- ❌ `inventory_data.db` - 資料庫檔案（不要上傳）
- ❌ `.streamlit/secrets.toml` - 敏感資訊（不要上傳）

## 🐛 常見問題

### 問題：應用程式無法連線到 Ragic API
**解決方案**：檢查 Streamlit Secrets 是否正確設定

### 問題：資料庫無法寫入
**解決方案**：Streamlit Cloud 的檔案系統是唯讀的，但 SQLite 資料庫可以正常運作（會儲存在臨時目錄）

### 問題：應用程式啟動失敗
**解決方案**：檢查 `requirements.txt` 是否包含所有必要的套件

## 📞 需要協助？

如果遇到問題，請檢查：
1. Streamlit Cloud 的日誌輸出
2. 應用程式的錯誤訊息
3. 確認所有必要的檔案都已上傳
