# -*- coding: utf-8 -*-
"""側欄：帳號、權限、同步狀態。"""

from __future__ import annotations

import streamlit as st


def render_sidebar_account(
    *,
    user: dict,
    role: str,
    roles: list[str],
    auth_verify,
    auth_change_password,
    auth_list_users,
    auth_create_user,
    auth_delete_user,
) -> None:
    st.sidebar.title("⚙️ 控制台")
    st.sidebar.caption(f"👤 {user['username']}（{role}）")
    st.sidebar.caption("📄 Google Sheet DB 同步功能已停用（僅保留 Ragic 流程）")

    if st.sidebar.button("🚪 登出", key="btn_logout"):
        del st.session_state["user"]
        st.rerun()

    with st.sidebar.expander("🔑 變更密碼", expanded=False):
        cur_p = st.text_input("目前密碼", type="password", key="chpwd_current")
        new_p1 = st.text_input("新密碼", type="password", key="chpwd_new1")
        new_p2 = st.text_input("確認新密碼", type="password", key="chpwd_new2")
        if st.button("💾 變更密碼", key="chpwd_btn"):
            u = auth_verify(user["username"], cur_p)
            if not u:
                st.error("目前密碼錯誤")
            elif not new_p1 or new_p1 != new_p2:
                st.error("新密碼不一致或為空")
            else:
                auth_change_password(user["username"], new_p1)
                st.success("已變更，請重新登入")
                del st.session_state["user"]
                st.rerun()

    if role == "行政主管":
        with st.sidebar.expander("👥 帳號管理", expanded=False):
            df_users = auth_list_users()
            st.dataframe(df_users[["username", "role"]], use_container_width=True, hide_index=True)
            st.caption("新增帳號")
            new_u = st.text_input("帳號", key="am_new_username", placeholder="username")
            new_p = st.text_input("密碼", type="password", key="am_new_password", placeholder="password")
            new_r = st.selectbox("權限", roles, key="am_new_role")
            if st.button("➕ 新增", key="am_btn_add"):
                ok, msg = auth_create_user(new_u, new_p, new_r)
                if ok:
                    st.success("已新增")
                    st.rerun()
                else:
                    st.error(msg)
            st.caption("刪除帳號")
            del_u = st.selectbox("選擇要刪除的帳號", df_users["username"].tolist(), key="am_del_user")
            if st.button("🗑️ 刪除", key="am_btn_del"):
                if del_u == user["username"]:
                    st.error("無法刪除目前登入的帳號")
                else:
                    auth_delete_user(del_u)
                    st.success("已刪除")
                    st.rerun()

