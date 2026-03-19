from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pandas as pd
import streamlit as st

from ragic_client import (
    RagicSheetRef,
    download_file,
    extract_entries,
    get_json,
    make_listing_url,
    parse_file_tokens,
    parse_sheet_url,
)


def render_ragic_test_tab(
    *,
    ragic_fields: dict[str, str],
    parse_cue_excel_for_table1: Callable[[bytes, Any], list[dict]],
) -> None:
    st.markdown("### 🧪 Ragic 抓取資料測試（導入前驗證用）")
    st.caption("可選擇抓取筆數/offset/日期區間，並顯示每筆專案欄位與 CUE Excel 解析結果（含波段拆分）。")

    # --- Debug log（可複製）---
    if "_ragic_debug_log" not in st.session_state:
        st.session_state["_ragic_debug_log"] = []

    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state["_ragic_debug_log"].append(f"[{ts}] {msg}")
        if len(st.session_state["_ragic_debug_log"]) > 500:
            st.session_state["_ragic_debug_log"] = st.session_state["_ragic_debug_log"][-500:]

    if not st.session_state["_ragic_debug_log"]:
        _log("頁面載入：尚未開始抓取（按「🚀 抓取並顯示」後會產生詳細 log）")

    default_ragic_url = "https://ap13.ragic.com/soundwow/forms12/17"
    ragic_url = st.text_input(
        "訂檔表單（Listing/Sheet）網址",
        value=default_ragic_url,
        help="格式類似：https://ap13.ragic.com/soundwow/forms12/17",
    )
    api_key_default = ""
    try:
        api_key_default = st.secrets.get("RAGIC_API_KEY", "")
    except Exception:
        api_key_default = ""
    api_key = st.text_input(
        "Ragic API Key",
        value=api_key_default,
        type="password",
        help="不會顯示內容；建議放在 .streamlit/secrets.toml 的 RAGIC_API_KEY",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        limit = st.number_input("抓取筆數 limit", min_value=1, max_value=2000, value=50, step=10)
    with c2:
        offset = st.number_input("起始 offset", min_value=0, max_value=200000, value=0, step=50)
    with c3:
        subtables0 = st.checkbox("不抓子表（subtables=0）", value=True)

    filter_field = st.selectbox(
        "日期篩選欄位（抓回後在本機篩）",
        options=["不篩", "建立日期", "執行開始日期", "執行結束日期"],
        index=0,
    )
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        date_from = st.date_input("日期起", value=None)
    with fcol2:
        date_to = st.date_input("日期迄", value=None)

    keyword = st.text_input("關鍵字（訂檔單號/客戶/產品/平台 任一包含）", value="")

    def g(e: dict, name: str) -> str:
        fid = ragic_fields.get(name)
        if not fid:
            return ""
        return e.get(fid, "")

    def entry_to_row(e: dict):
        cue_fid = ragic_fields.get("訂檔CUE表")
        cue_tokens = parse_file_tokens(e.get(cue_fid)) if cue_fid else []
        return {
            "_ragicId": e.get("_ragicId"),
            "訂檔單號": g(e, "訂檔單號"),
            "建立日期": g(e, "建立日期"),
            "客戶": g(e, "客戶"),
            "產品名稱": g(e, "產品名稱"),
            "平台": g(e, "平台"),
            "波段": g(e, "波段"),
            "總波段": g(e, "總波段"),
            "執行開始日期": g(e, "執行開始日期"),
            "執行結束日期": g(e, "執行結束日期"),
            "CUE表秒數": g(e, "CUE表秒數"),
            "CUE表總檔數": g(e, "CUE表總檔數"),
            "訂檔CUE表(檔案數)": len(cue_tokens),
        }

    def to_date(v):
        try:
            dt = pd.to_datetime(v, errors="coerce")
            return None if pd.isna(dt) else dt.date()
        except Exception:
            return None

    fetch_btn = st.button("🚀 抓取並顯示", type="primary")
    if fetch_btn:
        st.session_state["_ragic_debug_log"] = []
        _log("開始抓取")
        if not api_key.strip():
            st.error("請輸入 Ragic API Key（可放 .streamlit/secrets.toml 的 RAGIC_API_KEY）。")
            _log("API Key 為空，停止")
            st.stop()

        ref = parse_sheet_url(ragic_url)
        api_url = make_listing_url(ref, limit=int(limit), offset=int(offset), subtables0=subtables0, fts="")
        _log(f"API URL={api_url}")

        payload, err = get_json(api_url, api_key, timeout=60)
        if err:
            st.error(f"抓取失敗：{err}")
            _log(f"抓取失敗：{err}")
            st.stop()

        st.caption(f"API 回傳 keys 數量：{len(payload) if isinstance(payload, dict) else '非 dict'}")
        _log(f"payload type={type(payload).__name__}")

        if isinstance(payload, dict):
            try:
                k0 = next(iter(payload.keys()), None)
                st.markdown("#### 🔎 API 原始回傳（前 1 筆）")
                if k0 is not None:
                    st.json({str(k0): payload.get(k0)})
            except Exception:
                pass

        entries = extract_entries(payload)
        _log(f"entries count={len(entries)}")
        if not entries:
            st.warning("沒有抓到任何資料（可能沒有權限或資料為空）。")
            _log("entries 為空，停止")
            st.stop()

        try:
            e0 = entries[0]
            _log("第一筆 entry keys(head 20)=" + ",".join(list(e0.keys())[:20]))
            _log(f"第一筆 _ragicId={e0.get('_ragicId')}")
        except Exception as e:
            _log(f"第一筆 entry 診斷失敗：{e}")

        rows = [entry_to_row(e) for e in entries]
        df = pd.DataFrame(rows)
        _log(f"df rows={len(df)} cols={list(df.columns)}")
        if "_ragicId" in df.columns:
            _log(f"_ragicId notna count={int(df['_ragicId'].notna().sum())}")
            try:
                _log(f"_ragicId sample(head 5)={df['_ragicId'].head(5).tolist()}")
            except Exception:
                pass

        st.markdown("#### 🔎 解析後表格（前 5 列）")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)

        # 本機日期篩選
        if filter_field != "不篩" and (date_from or date_to):
            col = filter_field
            if col not in df.columns:
                st.warning(f"抓取結果沒有欄位「{col}」，已略過日期篩選。")
            else:
                df[col + "_date"] = df[col].apply(to_date)
                if date_from:
                    df = df[df[col + "_date"].notna() & (df[col + "_date"] >= date_from)]
                if date_to:
                    df = df[df[col + "_date"].notna() & (df[col + "_date"] <= date_to)]

        # 關鍵字篩選
        if keyword.strip():
            kw = keyword.strip()
            mask = False
            for c in ["訂檔單號", "客戶", "產品名稱", "平台"]:
                if c in df.columns:
                    mask = mask | df[c].astype(str).fillna("").str.contains(kw, regex=False)
            df = df[mask]

        st.markdown("#### ① 抓取結果（專案清單）")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### ② 檢視單筆專案 + 解析 CUE Excel")
        if "_ragicId" in df.columns:
            ids = pd.to_numeric(df["_ragicId"], errors="coerce").dropna().astype(int).tolist()
        else:
            ids = []
        if not ids:
            st.info("抓到的資料沒有 _ragicId，無法進一步解析。")
            _log("無可用 _ragicId")
            st.stop()

        sel_id = st.selectbox("選擇 _ragicId", options=ids)
        entry = next((e for e in entries if int(e.get("_ragicId", -1)) == int(sel_id)), None)
        if not entry:
            st.warning("找不到該筆 entry。")
            st.stop()

        show_fields = [
            "訂檔單號",
            "客戶",
            "產品名稱",
            "平台",
            "波段",
            "總波段",
            "執行開始日期",
            "執行結束日期",
            "CUE表秒數",
            "CUE表總檔數",
            "訂檔CUE表",
        ]
        info = {}
        for k in show_fields:
            fid = ragic_fields.get(k)
            info[k] = entry.get(fid) if fid else None
        st.json(info)

        cue_fid = ragic_fields.get("訂檔CUE表")
        cue_tokens = parse_file_tokens(entry.get(cue_fid)) if cue_fid else []
        if not cue_tokens:
            st.info("此筆沒有「訂檔CUE表」檔案。")
        else:
            st.markdown(f"**訂檔CUE表檔案數：{len(cue_tokens)}**")
            for i, tok in enumerate(cue_tokens, start=1):
                st.markdown(f"- 檔案{i}：`{tok}`")

            parse_now = st.checkbox("立即下載並解析 CUE Excel", value=True)
            if parse_now:
                for i, tok in enumerate(cue_tokens, start=1):
                    with st.expander(f"解析 檔案{i}"):
                        content, derr = download_file(ref, tok, api_key, timeout=120)
                        if derr:
                            st.error(f"下載失敗：{derr}")
                            _log(f"下載失敗 token={tok} err={derr}")
                            continue
                        cue_units = parse_cue_excel_for_table1(content, order_info=None)
                        st.markdown(f"解析出 ad_unit 筆數：**{len(cue_units)}**")
                        if cue_units:
                            df_units = pd.DataFrame(
                                [
                                    {
                                        "platform": u.get("platform"),
                                        "region": u.get("region"),
                                        "seconds": u.get("seconds"),
                                        "start_date": u.get("start_date"),
                                        "end_date": u.get("end_date"),
                                        "days": u.get("days"),
                                        "total_spots": u.get("total_spots"),
                                        "source_sheet": u.get("source_sheet"),
                                        "split_reason": u.get("split_reason"),
                                    }
                                    for u in cue_units
                                ]
                            )
                            st.dataframe(df_units, use_container_width=True, hide_index=True)

                            sample = []
                            for u in cue_units[:10]:
                                ds = u.get("daily_spots") or []
                                dts = u.get("dates") or []
                                sample.append(
                                    {
                                        "platform": u.get("platform"),
                                        "region": u.get("region"),
                                        "seconds": u.get("seconds"),
                                        "dates(head)": dts[:7],
                                        "daily_spots(head)": ds[:7],
                                    }
                                )
                            st.markdown("**每日檔次（前 7 天抽樣 / 前 10 筆）**")
                            st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)
                        else:
                            st.warning("此檔案沒有解析出每日檔次（可能不是預期版型或內容全空）。")

    st.markdown("---")
    st.markdown("#### 🧾 Debug Log（可直接複製貼回）")
    log_text = "\n".join(st.session_state.get("_ragic_debug_log", []))
    st.session_state["ragic_debug_log_area"] = log_text
    st.text_area("log", value=st.session_state.get("ragic_debug_log_area", ""), height=220, key="ragic_debug_log_area")
    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("清除 log", key="btn_clear_ragic_log"):
            st.session_state["_ragic_debug_log"] = []
            st.rerun()
    with b2:
        st.download_button(
            "下載 log.txt",
            data=(log_text or "").encode("utf-8"),
            file_name=f"ragic_debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_ragic_log",
        )

