# 本地 CUE 除錯工具使用說明

## 功能

1. **批量下載**：一次性從 Ragic 下載所有 CUE Excel 檔案到本地
2. **除錯標註**：查看原始 Excel、AI 解讀結果、進行人工標註
3. **匯出回饋**：將標註結果匯出為 JSON，方便回傳給開發者改進 AI 判斷邏輯

## 安裝

確保已安裝必要的套件：

```bash
# 方式 1：使用 requirements 檔案（推薦）
pip install -r local_debug_requirements.txt

# 方式 2：手動安裝
pip install streamlit pandas openpyxl requests
```

## 使用方式

### 1. 設定 API Key（可選）

如果不想使用預設的 API Key，可以設定環境變數：

```bash
# Windows
set RAGIC_API_KEY=your_api_key_here

# Linux/Mac
export RAGIC_API_KEY=your_api_key_here
```

### 2. 執行程式

```bash
streamlit run local_cue_debug_tool.py
```

程式會在瀏覽器中開啟（通常是 `http://localhost:8501`）

### 3. 下載 CUE 檔案

1. 在左側邊欄點擊「🔄 從 Ragic 下載所有 CUE 檔案」
2. 程式會自動下載所有包含 CUE 檔案的訂單
3. 已下載的檔案會儲存在 `local_cue_files/` 目錄
4. 如果檔案已存在，會自動跳過（避免重複下載）

### 4. 除錯標註

1. 在主畫面選擇要除錯的 CUE 檔案
2. 左側顯示原始 Excel 內容（可切換工作表）
3. 右側顯示 AI 解讀結果（可編輯標註）
4. 編輯完成後點擊「💾 儲存編輯結果」

### 5. 匯出回饋

1. 在除錯畫面點擊「💾 匯出回饋記錄」
2. 回饋記錄會儲存在 `feedback_records/` 目錄
3. 可以下載 JSON 檔案回傳給開發者

### 6. 批量匯出

在畫面底部可以點擊「📥 匯出所有回饋記錄」，一次匯出所有已標註的檔案

## 目錄結構

```
.
├── local_cue_debug_tool.py    # 主程式
├── local_cue_files/            # 下載的 CUE 檔案（自動建立）
├── feedback_records/           # 匯出的回饋記錄（自動建立）
└── ragic_inventory.py          # 主程式（用於導入解析函數）
```

## 回饋記錄格式

匯出的 JSON 檔案包含：

```json
{
  "file_hash": "檔案 MD5 hash",
  "file_name": "檔案名稱",
  "rid": "Ragic 記錄 ID",
  "created_at": "建立時間",
  "ai_interpretations": [
    {
      "sheet": "工作表名稱",
      "row_idx": 行號,
      "spots": 檔次,
      "confidence": "信心等級",
      "rule_used": "使用的規則",
      "reason": "判斷原因",
      "status": "狀態（pending/accepted/rejected/corrected）",
      "human_note": "人工備註"
    }
  ],
  "table1_split_data": [...],
  "human_corrections": [...]
}
```

## 注意事項

1. **首次使用**：需要先下載 CUE 檔案才能進行除錯
2. **檔案管理**：下載的檔案會保留在本地，可以隨時查看
3. **回饋記錄**：建議定期匯出回饋記錄，避免資料遺失
4. **API Key**：如果下載失敗，請檢查 API Key 是否正確

## 與主程式的關係

- 本地除錯工具會從 `ragic_inventory.py` 導入解析函數
- 確保 `ragic_inventory.py` 在同一目錄下
- 當 AI 判斷邏輯改進後，可以直接更新 `ragic_inventory.py`，本地工具會自動使用新邏輯

## 工作流程

1. **開發階段**：使用本地工具進行除錯和標註
2. **回饋階段**：匯出回饋記錄給開發者
3. **改進階段**：開發者根據回饋改進 AI 判斷邏輯
4. **測試階段**：在本地工具中測試改進後的邏輯
5. **整合階段**：確認無誤後，將改進的邏輯合併到 Streamlit 主程式
