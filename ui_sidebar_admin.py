# -*- coding: utf-8 -*-
"""側欄：管理工具與平台設定。"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def render_sidebar_admin(
    *,
    get_db_connection: Callable[[], object],
    init_db: Callable[[], None],
    db_file: str,
    get_store_count: Callable[[str, dict], int],
    load_platform_settings: Callable[[], dict],
    platform_capacity: dict,
    save_platform_settings: Callable[[str, int, int], None],
    sync_sheets_if_enabled: Callable[..., object],
) -> None:
    st.sidebar.markdown("---")
    if st.sidebar.button("🧨 重置資料庫（清空資料，保留 Users）", help="⚠️ 警告：會清空主要業務資料，保留帳號權限"):
        try:
            init_db()
            conn = get_db_connection()
            try:
                c = conn.cursor()
                c.execute("DELETE FROM orders")
                c.execute("DELETE FROM ad_flight_segments")
                c.execute("DELETE FROM platform_settings")
                c.execute("DELETE FROM platform_monthly_capacity")
                c.execute("DELETE FROM platform_monthly_purchase")
                c.execute("DELETE FROM ragic_import_logs")
                # 保留 users（帳號權限不動）
                conn.commit()
            finally:
                conn.close()

            st.sidebar.success("✅ 已清空資料庫資料（Users 保留）")
        except Exception as e:
            st.sidebar.error(f"❌ 刪除失敗: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 平台設定")
    with st.sidebar.expander("設定平台店數與營業時間"):
        conn = get_db_connection()
        platforms = pd.read_sql("SELECT DISTINCT platform FROM orders", conn)
        conn.close()

        if not platforms.empty:
            custom_settings = load_platform_settings()
            sel_platform = st.selectbox("選擇平台", platforms["platform"].tolist())
            current_store = get_store_count(sel_platform, custom_settings)
            current_hours = platform_capacity.get(sel_platform, 18)
            if custom_settings and sel_platform in custom_settings:
                current_hours = custom_settings[sel_platform]["daily_hours"]

            new_store = st.number_input("店數", min_value=1, value=int(current_store), step=1)
            new_hours = st.number_input("每日營業小時數", min_value=1, max_value=24, value=int(current_hours), step=1)

            if st.button("💾 儲存設定"):
                save_platform_settings(sel_platform, new_store, new_hours)
                st.success("設定已儲存！")
                st.rerun()
        else:
            st.info("請先新增訂單或匯入資料")

