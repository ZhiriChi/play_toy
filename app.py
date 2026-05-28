"""Streamlit UI: chat + material upload + monitoring dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from llm_local import Pipeline
from llm_local.monitoring import (
    by_model_breakdown,
    cluster_topics,
    daily_volume,
    high_uncertainty_segments,
    quality_metrics,
    top_tokens,
    week_ago_ts,
)

st.set_page_config(page_title="本地 LLM · 三层架构", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="初始化模型管道…")
def get_pipeline() -> Pipeline:
    return Pipeline()


pipe = get_pipeline()


def _entropy_color(value: float, vmax: float) -> str:
    if vmax <= 0:
        return "#10b981"
    ratio = min(value / vmax, 1.0)
    # green -> yellow -> red
    r = int(255 * ratio)
    g = int(180 * (1 - ratio) + 50)
    b = 50
    return f"rgb({r},{g},{b})"


def render_token_heatmap(tokens: list[dict]) -> None:
    ents = [t["entropy"] for t in tokens if t.get("entropy") is not None]
    if not ents:
        st.caption("此次推理没有捕获到 token logprobs(检查 Ollama 版本是否支持)。")
        return
    vmax = max(ents)
    html_parts = ["<div style='line-height:2.4; font-family: monospace;'>"]
    for t in tokens:
        ent = t.get("entropy")
        tok = (t.get("token") or "").replace("<", "&lt;").replace(">", "&gt;")
        if not tok.strip():
            tok = "·"
        color = _entropy_color(ent, vmax) if ent is not None else "#94a3b8"
        title = f"H={ent:.2f}, logp={t.get('chosen_logprob'):.2f}" if ent is not None else ""
        html_parts.append(
            f"<span title='{title}' style='background:{color}22; "
            f"border-bottom:2px solid {color}; padding:1px 3px; margin:1px; border-radius:3px;'>{tok}</span>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_entropy_chart(tokens: list[dict]) -> None:
    df = pd.DataFrame([
        {"position": t["position"], "token": t["token"], "entropy": t.get("entropy") or 0,
         "logprob": t.get("chosen_logprob")}
        for t in tokens
    ])
    if df.empty:
        return
    fig = px.bar(
        df, x="position", y="entropy",
        hover_data=["token", "logprob"],
        color="entropy",
        color_continuous_scale="RdYlGn_r",
        title="每个 token 的熵 (entropy)",
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, conv_id, meta}

available_models = pipe.available_models()
model_label_to_name = {m["label"]: m["name"] for m in available_models}
default_model = pipe.cfg["ollama"]["default_model"]
default_label = next(
    (m["label"] for m in available_models if m["name"] == default_model),
    available_models[0]["label"],
)
if "model_label" not in st.session_state:
    st.session_state.model_label = default_label

tab_chat, tab_docs, tab_dash = st.tabs(["💬 对话", "📁 材料库", "📊 监控 Dashboard"])

# ============ Chat tab ============
with tab_chat:
    st.title("本地 LLM · 三层架构")

    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.caption(
            f"嵌入: `{pipe.cfg['ollama']['embed_model']}` · 完全离线 · "
            f"切换模型不会清空对话历史"
        )
    with top_r:
        st.session_state.model_label = st.selectbox(
            "🤖 当前模型",
            list(model_label_to_name.keys()),
            index=list(model_label_to_name.keys()).index(st.session_state.model_label),
        )
    active_model = model_label_to_name[st.session_state.model_label]
    active_info = pipe.model_info(active_model) or {}

    docs = pipe.list_documents()
    if not docs:
        st.warning("尚未上传任何材料。可以直接提问(只走 Layer 1),也可以去『材料库』上传文件以启用 Layer 2。")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                conv_id = msg.get("conv_id")
                if meta.get("thinking"):
                    with st.expander("🧠 思考过程 (Reasoning)"):
                        st.markdown(meta["thinking"])
                cols = st.columns([1, 1, 1, 1, 2])
                cols[0].metric("模型", meta.get("model", "—").split(":")[0])
                cols[1].metric("平均熵", f"{meta['avg_entropy']:.2f}" if meta.get('avg_entropy') is not None else "—")
                cols[2].metric("交叉熵", f"{meta['cross_entropy']:.2f}" if meta.get('cross_entropy') is not None else "—")
                cols[3].metric("延迟", f"{meta['latency_ms']:.0f} ms")
                with cols[4]:
                    if conv_id:
                        clear_key = f"clear_{conv_id}"
                        if st.button("✅ 我清楚理解了", key=clear_key, help="奖励本次用到的材料块"):
                            r = pipe.mark_clear(conv_id, True)
                            st.success(f"已奖励 {len(r.get('rewarded_chunks', []))} 个材料块。")

                with st.expander("🔍 命中材料"):
                    for h in meta.get("hits", []):
                        src = (h.get("metadata") or {}).get("filename", "?")
                        st.markdown(
                            f"**{src}** · sim={h['similarity']:.3f} · "
                            f"reward={h['reward_score']:.1f} · final={h['final_score']:.3f}"
                        )
                        st.code(h["content"][:500], language=None)

                with st.expander("🌡️ Token 熵 热力图"):
                    render_token_heatmap(meta.get("tokens", []))
                    render_entropy_chart(meta.get("tokens", []))

    query = st.chat_input(f"提问…(当前: {st.session_state.model_label})")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            spinner_text = "推理中…" if active_info.get("reasoning") else "思考中…"
            with st.spinner(spinner_text):
                try:
                    out = pipe.ask(query, model=active_model)
                except Exception as e:
                    st.error(f"调用失败: {e}")
                    st.stop()
            if out.get("thinking"):
                with st.expander("🧠 思考过程 (Reasoning)"):
                    st.markdown(out["thinking"])
            st.markdown(out["response"])
            meta = {
                "model": out["model"],
                "thinking": out.get("thinking"),
                "avg_entropy": out["avg_entropy"],
                "cross_entropy": out["cross_entropy"],
                "latency_ms": out["latency_ms"],
                "hits": out["hits"],
                "tokens": out["tokens"],
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": out["response"],
                "conv_id": out["conversation_id"],
                "meta": meta,
            })
            st.rerun()

# ============ Documents tab ============
with tab_docs:
    st.header("材料库 — Layer 2")
    st.caption("上传 TXT/MD/PDF/DOCX。扫描版 PDF 会自动 OCR 识别。文件只在本地处理,不会出网。")

    uploads = st.file_uploader(
        "上传材料",
        type=["txt", "md", "markdown", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploads:
        save_dir = Path(pipe.cfg["paths"]["uploads_dir"])
        for f in uploads:
            target = save_dir / f.name
            target.write_bytes(f.getvalue())
            with st.spinner(f"处理 {f.name} …"):
                try:
                    info = pipe.ingest(target)
                    if info.get("n_chunks", 0) == 0:
                        st.warning(
                            f"⚠️ {f.name}: 0 个片段 — 文件没有抽出任何文字,"
                            "可能是空文档。"
                        )
                    else:
                        msg = f"✅ {f.name}: {info['n_chunks']} 个片段已索引"
                        if info.get("ocr_pages"):
                            msg += f"(其中 {info['ocr_pages']} 页通过 OCR 识别)"
                        st.success(msg)
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")

    st.divider()
    docs = pipe.list_documents()
    if not docs:
        st.info("还没有材料。")
    else:
        df = pd.DataFrame(docs)
        df["uploaded_at"] = df["uploaded_at"].apply(
            lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M")
        )
        for _, row in df.iterrows():
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            c1.write(f"**{row['filename']}**")
            c2.caption(row["uploaded_at"])
            c3.caption(f"{row['n_chunks']} 片段")
            if c4.button("删除", key=f"del_{row['id']}"):
                pipe.delete_document(row["id"])
                st.rerun()

# ============ Dashboard tab ============
with tab_dash:
    st.header("监控 Dashboard")

    range_label = st.selectbox("时间范围", ["近 7 天", "近 30 天", "全部"], index=0)
    if range_label == "近 7 天":
        since = week_ago_ts(1)
    elif range_label == "近 30 天":
        since = week_ago_ts(4)
    else:
        since = 0.0

    metrics = quality_metrics(pipe.storage, since_ts=since)
    cols = st.columns(5)
    cols[0].metric("问答总数", metrics.get("n", 0))
    cols[1].metric(
        "平均 token 熵",
        f"{metrics['avg_entropy']:.3f}" if metrics.get("avg_entropy") is not None else "—",
        help="越低 = 模型越笃定",
    )
    cols[2].metric(
        "平均交叉熵",
        f"{metrics['avg_cross_entropy']:.3f}" if metrics.get("avg_cross_entropy") is not None else "—",
        help="对生成序列的负对数似然均值",
    )
    cols[3].metric(
        "理解清晰率",
        f"{metrics['clarity_rate']*100:.0f}%" if metrics.get("clarity_rate") is not None else "—",
        help="用户标记『清楚理解』的占比",
    )
    cols[4].metric(
        "平均延迟",
        f"{metrics['avg_latency_ms']:.0f} ms" if metrics.get("avg_latency_ms") is not None else "—",
    )

    by_model = by_model_breakdown(pipe.storage, since_ts=since)
    if by_model:
        st.subheader("🤖 按模型对比")
        df_bm = pd.DataFrame([
            {
                "模型": r["model"],
                "问答数": r["n"],
                "平均熵": round(r["avg_entropy"], 3) if r["avg_entropy"] is not None else None,
                "平均交叉熵": round(r["avg_cross_entropy"], 3) if r["avg_cross_entropy"] is not None else None,
                "平均延迟 (ms)": round(r["avg_latency_ms"], 0) if r["avg_latency_ms"] is not None else None,
            }
            for r in by_model
        ])
        st.dataframe(df_bm, use_container_width=True, hide_index=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🗣️ 单次问答的 token 熵")
        recents = pipe.storage.recent_conversations(limit=50)
        if not recents:
            st.info("暂无历史对话。先去『对话』tab 提几个问题。")
        else:
            options = {
                f"#{c['id']} · {datetime.fromtimestamp(c['ts']).strftime('%m-%d %H:%M')} · "
                f"[{(c.get('model') or '?').split(':')[0]}] · {c['query'][:30]}": c["id"]
                for c in recents
            }
            picked = st.selectbox("选择一次对话", list(options.keys()))
            conv = pipe.storage.get_conversation(options[picked])
            if conv:
                c1, c2, c3 = st.columns(3)
                c1.metric("model", (conv.get("model") or "—").split(":")[0])
                c2.metric("avg entropy", f"{conv['avg_entropy']:.3f}" if conv.get("avg_entropy") is not None else "—")
                c3.metric("cross entropy", f"{conv['cross_entropy']:.3f}" if conv.get("cross_entropy") is not None else "—")
                render_entropy_chart(conv.get("tokens") or [])
                with st.expander("高不确定性 token (top 20%)"):
                    seg = high_uncertainty_segments(conv, top_pct=0.2)
                    if seg:
                        st.dataframe(pd.DataFrame(seg), use_container_width=True, hide_index=True)
                    else:
                        st.caption("无可用 entropy 数据。")
                with st.expander("回答全文 + 热力图"):
                    st.markdown(conv["response"])
                    render_token_heatmap(conv.get("tokens") or [])

    with right:
        st.subheader("📅 每日提问量")
        vol = daily_volume(pipe.storage, days=14)
        df = pd.DataFrame(vol)
        fig = px.line(df, x="date", y="count", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 高频 token (问题中)")
        toks = top_tokens(pipe.storage, since_ts=since, n=20)
        if toks:
            df_t = pd.DataFrame(toks, columns=["token", "count"])
            fig = px.bar(df_t, x="count", y="token", orientation="h")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("数据不足。")

    st.divider()

    st.subheader("🧭 本周热门 topic (自动聚类)")
    if st.button("重新计算 topic 聚类", help="对历史问题做 KMeans"):
        st.session_state["_topics"] = None
    if "_topics" not in st.session_state or st.session_state["_topics"] is None:
        try:
            with st.spinner("聚类中…"):
                topics = cluster_topics(
                    pipe.storage, pipe.client,
                    embed_model=pipe.cfg["ollama"]["embed_model"],
                    since_ts=since,
                )
                st.session_state["_topics"] = topics
        except Exception as e:
            st.warning(f"聚类失败: {e}")
            st.session_state["_topics"] = []
    topics = st.session_state.get("_topics") or []
    if topics:
        df_topic = pd.DataFrame(topics)
        fig = go.Figure(go.Pie(labels=df_topic["label"], values=df_topic["count"], hole=0.4))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        for t in topics:
            with st.expander(f"{t['label']}  ·  {t['count']} 条"):
                for s in t.get("samples", []):
                    st.write(f"• {s}")
    else:
        st.caption("数据不足以聚类(至少 3 条问答)。")

    st.divider()
    st.subheader("🏆 被奖励最多的材料片段 (Layer 3 → Layer 2 反馈)")
    top_rewarded = pipe.storage.top_chunks_by_reward(limit=10)
    if top_rewarded:
        df_r = pd.DataFrame([
            {
                "文件": r["filename"],
                "reward": r["reward_score"],
                "用过": r["n_used"],
                "被奖励": r["n_rewarded"],
                "片段(节选)": (r["content"][:120] + "…") if len(r["content"]) > 120 else r["content"],
            }
            for r in top_rewarded
        ])
        st.dataframe(df_r, use_container_width=True, hide_index=True)
    else:
        st.caption("还没有被奖励的材料片段。回答之后点『我清楚理解了』会奖励对应材料。")

    with st.expander("📚 其他业界常用 LLM 监控维度(已实现 + 路线图)"):
        st.markdown(
            """
**本仪表已支持**
- Token-level **entropy / cross-entropy**: 模型对每一步的不确定性 + 对整段输出的负对数似然
- **High-uncertainty 区段**: 自动找出最不确定的 token,辅助定位幻觉风险
- **Retrieval quality**: 命中材料的 similarity & reward 分数(每次对话展开可见)
- **Clarity rate**: 用户反馈"清楚理解"的占比 — 业界 thumbs-up rate 的等价物
- **Topic clustering / drift**: 对历史 query 做 KMeans 聚类,观察主题分布变化
- **Token frequency**: 最常被问到的关键词
- **Latency / volume**: 每次延迟 + 每日提问量趋势
- **Reward propagation**: Layer 3 的反馈如何反向加权 Layer 2 的检索

**业界其他常用维度(可后续加)**
- *Self-consistency*: 同一问题多次采样,看答案稳定度
- *SelfCheckGPT*: 用模型自身重述答案核对一致性,作为幻觉指标
- *Toxicity / PII scan*: 对回答做安全分类(本地小模型即可)
- *Embedding drift*: 监控 query embedding 的均值随时间漂移
- *Cache hit / cost*: token throughput、GPU 利用率
- *A/B*: 对比不同 system prompt 或不同 reward 权重的效果
            """
        )
