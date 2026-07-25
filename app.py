"""Punch.io — Streamlit demo with live RAG.

The self-hostable product is the CLI (sync.py) + React frontend. This Streamlit app
is the zero-setup DEMO: onboard a Discord bot, seed/sync messages, rebuild the FAISS
index live in the browser, and chat with RAG over your project communications.

Run:  uv run streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from core import store
from core.connectors.discord import DiscordConnector
from ingest.pipeline import sync_source

st.set_page_config(page_title="Punch.io", page_icon="🥊", layout="wide")

st.markdown("""
<style>
    .main .block-container { max-width: 1200px; padding: 1rem 2rem !important; }
    .stApp { background: #0a0a0c; }
    .stApp header { background: #0a0a0c !important; }
    h1, h2, h3 { font-weight: 500; letter-spacing: -0.02em; color: #e0e0e0; }
    p, li, .stMarkdown { color: #b0b0b0; }

    /* sidebar */
    section[data-testid="stSidebar"] { background: #0d0d10; border-right: 1px solid #1a1a1e; }
    section[data-testid="stSidebar"] .stButton button { width: 100%; text-align: left; border: none; border-bottom: 1px solid #1a1a1e; background: transparent; border-radius: 0; padding: 0.6rem 0; color: #888; font-size: 0.8rem; }
    section[data-testid="stSidebar"] .stButton button:hover { color: #e0e0e0; background: transparent; }
    section[data-testid="stSidebar"] .stMetric { background: transparent; border: 1px solid #1a1a1e; padding: 0.6rem; }
    section[data-testid="stSidebar"] .stMetric label { color: #666; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { font-size: 1.2rem; color: #e0e0e0; }

    /* buttons */
    .stButton button { font-weight: 400; border-radius: 4px; border: 1px solid #1e1e22; background: transparent; color: #aaa; padding: 0.3rem 0.8rem; font-size: 0.8rem; transition: all 0.15s; }
    .stButton button:hover { border-color: #444; color: #e0e0e0; }
    .stButton button[kind="primary"] { background: #e0e0e0; color: #0a0a0c; border: none; font-weight: 500; }
    .stButton button[kind="primary"]:hover { background: #fff; }

    /* nav tabs */
    .stColumns .stButton button { border: none; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; color: #666; padding: 0.5rem 0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .stColumns .stButton button[kind="primary"] { border-bottom-color: #e0e0e0; color: #e0e0e0; background: transparent; }
    .stColumns .stButton button[kind="primary"]:hover { background: transparent; }
    .stColumns .stButton button:hover { color: #e0e0e0; }

    /* chat */
    .stChatMessage { background: transparent; padding: 0.5rem 0; }
    .stChatMessage [data-testid="chatAvatarIcon"] { width: 28px; height: 28px; font-size: 0.7rem; }
    div[data-testid="chatMessageContent"] { background: #131317; border: 1px solid #1a1a1e; border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; color: #d0d0d0; }
    .stChatMessage[data-testid="userChatMessage"] div[data-testid="chatMessageContent"] { background: #1a1a22; }
    .stChatInput { border-radius: 8px; border-color: #1e1e22; background: #131317; }
    .stChatInput:focus { border-color: #444; }

    /* inputs */
    .stSelectbox > div, .stMultiselect > div, .stTextInput > div > input { border-radius: 4px; border-color: #1e1e22; background: #131317; color: #d0d0d0; font-size: 0.8rem; }
    .stSelectbox > div:focus, .stMultiselect > div:focus, .stTextInput > div > input:focus { border-color: #444; }

    /* metric cards */
    .stMetric { background: transparent; border: 1px solid #1a1a1e; padding: 0.8rem; border-radius: 4px; }
    .stMetric label { color: #666; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem; color: #e0e0e0; }

    /* alerts / info */
    .element-container .stAlert { background: #111114; border: 1px solid #1a1a1e; border-radius: 4px; color: #b0b0b0; font-size: 0.85rem; }
    .element-container .stAlert[data-baseweb="notification"] { background: #111114; }

    /* containers / borders */
    div[data-testid="stExpander"] { border: 1px solid #1a1a1e; border-radius: 4px; }
    div.stForm { border: 1px solid #1a1a1e; border-radius: 8px; padding: 1.5rem; background: #0d0d10; }

    /* progress */
    .stProgress > div > div { border-radius: 2px; }
    .stProgress > div { background: #1a1a1e; }

    /* misc */
    hr { border-color: #1a1a1e; margin: 1.5rem 0; }
    .stCaption { color: #555; font-size: 0.75rem; }
    .st-spinner { color: #666; }
    div[data-testid="stNotification"] { background: #111114; border: 1px solid #1a1a1e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "punch.db")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_db", "index.faiss")


# --- small helpers -----------------------------------------------------------

def _db_ready() -> bool:
    return os.path.exists(DB_PATH)


def _index_ready() -> bool:
    return os.path.exists(FAISS_PATH)


def _load_messages(project: str | None = None):
    """Read messages straight from SQLite. Tiny dataset -> no caching needed."""
    if not _db_ready():
        return []
    return store.get_messages(db_path=DB_PATH, project=project)


def _rebuild_index() -> int:
    """Live RAG build: embed unembedded messages into FAISS. Returns count embedded.

    Resets generation/llm.py's cached retriever so chat picks up the fresh index.
    """
    from ingest.pipeline import embed_new
    from generation import llm as llm_module

    n = embed_new(db_path=DB_PATH, faiss_dir=os.path.join(DATA_DIR, "faiss_db"))
    # ponytail: the retriever is a module global cached on first query; without this
    # reset a rebuild would keep serving the stale index for the rest of the session.
    llm_module._retriever = None
    llm_module._vectorstore = None
    return n


# --- onboarding (Discord bot setup) ------------------------------------------

def _validate_token(token: str) -> tuple[bool, str]:
    """Hit Discord's /users/@me with the token. Cheap, authoritative check."""
    try:
        r = requests.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {token}"},
            timeout=10,
        )
    except Exception as e:
        return False, f"Could not reach Discord: {e}"
    if r.status_code == 200:
        return True, r.json().get("username", "your bot")
    if r.status_code == 401:
        return False, "Discord rejected this token (401). Copy it again from the Bot page."
    return False, f"Unexpected response ({r.status_code})."


def render_onboarding():
    st.title("Punch.io")
    st.caption(
        "Connect a Discord bot to pull your team's messages into a searchable "
        "project brain. Set up once, then ask questions about your work."
    )

    steps = [
        (
            "Create a Discord application",
            "Open the [Discord Developer Portal](https://discord.com/developers/applications), "
            "click **New Application**, and name it.",
        ),
        (
            "Add a bot and copy its token",
            "In the **Bot** tab, click **Reset Token**, copy it, "
            "and paste it into the form below.",
        ),
        (
            "Enable Message Content Intent",
            "Still in the **Bot** tab, turn on **Message Content Intent** "
            "under **Privileged Gateway Intents**."
        ),
        (
            "Invite the bot to your server",
            "In **OAuth2 > URL Generator**, check **bot**, then "
            "**Read Messages/View Channels** and **Read Message History**. "
            "Open the generated URL and authorize."
        ),
    ]

    for i, (title, body) in enumerate(steps):
        with st.container(border=True):
            cols = st.columns([1, 10])
            cols[0].markdown(f"<span style='color:#555;font-size:0.65rem;font-weight:600'>{i+1:02d}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"**{title}**")
            cols[1].markdown(body)

    st.divider()
    st.markdown("**Credentials**")

    with st.form("creds"):
        token = st.text_input("Discord bot token", type="password",
                              value=os.getenv("DISCORD_BOT_TOKEN", ""))
        guild = st.text_input("Server ID (optional — blank auto-discovers all servers)",
                             value=os.getenv("DISCORD_GUILD_ID", ""),
                             help="Right-click your server icon > Copy Server ID "
                                  "(enable Developer Mode in Discord settings first). "
                                  "Leave blank to auto-discover channels from every server "
                                  "the bot can see.")
        groq_key = st.text_input("Groq API key", type="password",
                                value=os.getenv("GROQ_API_KEY", ""),
                                help="Required for Q&A. Get one at https://console.groq.com/keys")
        submitted = st.form_submit_button("Connect", type="primary")

    if submitted:
        if not token:
            st.error("Discord bot token is required.")
            return
        if not groq_key:
            st.error("Groq API key is required.")
            return
        ok, msg = _validate_token(token)
        if ok:
            st.session_state["discord_token"] = token
            st.session_state["discord_guild_id"] = guild
            st.session_state["groq_api_key"] = groq_key
            st.session_state["onboarded"] = True
            st.session_state["force_onboard"] = False
            st.success(f"Connected as {msg}.")
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.info(
        "Data already synced from the CLI? Skip this setup — Punch.io will "
        "use your existing database automatically."
    )


# --- channel discovery & sync -------------------------------------------------

DISCORD_API = "https://discord.com/api/v10"


def _discord_get(path: str, token: str):
    url = f"{DISCORD_API}{path}"
    headers = {"Authorization": f"Bot {token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 429:
        time.sleep(float(resp.json().get("retry_after", 1)) + 0.25)
        resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _discover_guild_channels(token: str, guild_id: str | None = None) -> list[dict]:
    if guild_id:
        channels = _discord_get(f"/guilds/{guild_id}/channels", token)
        return [c for c in channels if c.get("type") == 0]
    guilds = _discord_get("/users/@me/guilds", token)
    all_channels = []
    for g in guilds:
        try:
            channels = _discord_get(f"/guilds/{g['id']}/channels", token)
            all_channels.extend(c for c in channels if c.get("type") == 0)
        except requests.HTTPError:
            continue
    return all_channels


def render_channel_setup():
    st.title("Select channels to sync")
    st.caption("Choose which Discord channels to pull messages from.")

    token = st.session_state.get("discord_token") or os.environ.get("DISCORD_BOT_TOKEN")
    guild_id = st.session_state.get("discord_guild_id") or os.environ.get("DISCORD_GUILD_ID")

    if "available_channels" not in st.session_state:
        with st.spinner("Discovering channels..."):
            try:
                st.session_state.available_channels = _discover_guild_channels(token, guild_id)
            except Exception as e:
                st.error(f"Failed to discover channels: {e}")
                st.stop()

    channels = st.session_state.available_channels
    if not channels:
        st.warning("No text channels found. Make sure the bot is invited to the server with proper permissions.")
        if st.button("Back to setup"):
            st.session_state.force_onboard = True
            st.session_state.onboarded = False
            st.rerun()
        return

    names = [c["name"] for c in channels]
    selected = st.multiselect("Channels", names, default=names)

    if st.button("Sync selected channels", type="primary", width="stretch"):
        progress = st.progress(0, "Syncing...")
        results = {}
        for i, ch in enumerate(channels):
            if ch["name"] not in selected:
                continue
            progress.progress((i) / len(selected), f"Syncing #{ch['name']}...")
            connector = DiscordConnector(
                channel=ch["name"],
                channel_id=ch["id"],
                bot_token=token,
            )
            try:
                summary = sync_source(connector, db_path=DB_PATH, faiss_dir=os.path.join(DATA_DIR, "faiss_db"))
                results[ch["name"]] = f"received {summary['received']}, new {summary['new']}"
            except Exception as e:
                results[ch["name"]] = f"error: {e}"
        progress.progress(1.0, "Building search index...")

        _rebuild_index()
        st.session_state.synced_channels = [c for c in channels if c["name"] in selected]
        st.session_state.channels_setup = True
        progress.empty()

        st.success("Sync complete!")
        for name, msg in results.items():
            st.write(f"- #{name}: {msg}")
        st.rerun()


# --- timeline extraction (LLM-powered) ---------------------------------------

EVENT_TYPES = {"decision", "milestone", "blocker", "resolution"}

def _extract_system() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    return f"""Today is {now}. You analyze a project team's chat log and extract the timeline of \
what actually happened. Output ONLY events that matter to a project manager: decisions \
made, milestones reached, blockers raised, and blockers resolved. Ignore routine chatter.

Return a JSON array. Each event is an object:
{{"date": "YYYY-MM-DD", "type": "decision|milestone|blocker|resolution",
 "summary": "one concise sentence", "channel": "<channel>"}}

Rules:
- type MUST be exactly one of: decision, milestone, blocker, resolution.
- date MUST be the message's date (given per line).
- summary is one sentence, concrete, no fluff.
- A "resolution" resolves an earlier "blocker".
- Output the JSON array and NOTHING else. No markdown fences, no prose."""


def _messages_block(messages) -> str:
    lines = []
    for m in messages:
        date = (m.timestamp or "")[:10]
        lines.append(f"{date} | {m.channel} | {m.author}: {m.content}")
    return "\n".join(lines)


def _parse_events(raw_text: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        events = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    clean = []
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("type") not in EVENT_TYPES:
            continue
        if not e.get("date") or not e.get("summary"):
            continue
        clean.append({
            "date": str(e["date"])[:10],
            "type": e["type"],
            "summary": str(e["summary"]),
            "channel": str(e.get("channel", "")),
        })
    clean.sort(key=lambda e: e["date"])
    return clean


def _extract_timeline(messages) -> list[dict]:
    from langchain_core.messages import SystemMessage, HumanMessage
    from generation.llm import get_llm
    resp = get_llm(st.session_state.get("groq_api_key")).invoke([
        SystemMessage(content=_extract_system()),
        HumanMessage(content=f"Project chat log:\n\n{_messages_block(messages)}"),
    ])
    return _parse_events(resp.content)


# --- chart helpers -----------------------------------------------------------

def _blocker_trend(events):
    by_date = {}
    for e in events:
        if e["type"] not in ("blocker", "resolution"):
            continue
        row = by_date.setdefault(e["date"], {"date": e["date"], "opened": 0, "resolved": 0})
        if e["type"] == "blocker":
            row["opened"] += 1
        else:
            row["resolved"] += 1
    dates = sorted(by_date.values(), key=lambda r: r["date"])
    open_running = 0
    resolved_running = 0
    result = []
    for r in dates:
        open_running += r["opened"] - r["resolved"]
        resolved_running += r["resolved"]
        result.append({"date": r["date"], "open": max(0, open_running), "resolvedTotal": resolved_running})
    df = pd.DataFrame(result)
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def _activity_by_channel(messages):
    by_date = defaultdict(lambda: defaultdict(int))
    for m in messages:
        date = (m.timestamp or "")[:10]
        if date:
            by_date[date][m.channel] += 1
    dates = sorted(by_date)
    channels = sorted({m.channel for m in messages})
    data = {c: [by_date[d].get(c, 0) for d in dates] for c in channels}
    df = pd.DataFrame(data, index=pd.Index(dates, name="date"))
    return df


# --- timeline & chart view ---------------------------------------------------

def render_timeline_tab(messages):
    st.subheader("Activity")
    st.caption("Messages per channel per day — no LLM needed.")
    activity = _activity_by_channel(messages)
    if not activity.empty:
        st.area_chart(activity)
    else:
        st.caption("No message data to chart.")

    st.divider()
    st.subheader("Timeline")
    st.caption("Decisions, milestones, blockers, and resolutions extracted from messages.")

    if "timeline_events" not in st.session_state:
        st.session_state.timeline_events = None

    if st.button("Extract timeline from messages",
                 help="Uses your LLM to extract decisions, milestones, blockers, and resolutions.",
                 width="stretch"):
        with st.spinner("Analyzing messages..."):
            try:
                events = _extract_timeline(messages)
                st.session_state.timeline_events = events
                st.rerun()
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    events = st.session_state.timeline_events
    if events is None:
        return

    cols = st.columns(4)
    cols[0].metric("Events", len(events))
    counts = {"blocker": 0, "decision": 0, "milestone": 0, "resolution": 0}
    for e in events:
        if e["type"] in counts:
            counts[e["type"]] += 1
    cols[1].metric("Decisions", counts["decision"])
    cols[2].metric("Blockers", counts["blocker"])
    cols[3].metric("Resolutions", counts["resolution"])

    if counts["blocker"] > 0 or counts["resolution"] > 0:
        st.subheader("Blockers")
        trend = _blocker_trend(events)
        if not trend.empty:
            st.line_chart(trend)

    import html as _html
    import streamlit.components.v1 as _components
    df_tl = pd.DataFrame(events)
    df_tl["date"] = pd.to_datetime(df_tl["date"])
    df_tl = df_tl.sort_values("date").reset_index(drop=True)
    _colors = {"decision": "#4C9BE8", "milestone": "#4CAF50", "blocker": "#E85454", "resolution": "#FFA726"}
    n = len(df_tl)
    rows_html = ""
    for i, row in df_tl.iterrows():
        c = _colors.get(row["type"], "#888")
        summary = _html.escape(row["summary"][:70]) + ("…" if len(row["summary"]) > 70 else "")
        date_str = row["date"].strftime("%b %d, %Y")
        rows_html += f"""<tr>
          <td style="width:110px;text-align:right;padding:6px 10px 6px 0;font-size:.7rem;color:#888;white-space:nowrap">{date_str}</td>
          <td style="width:20px;text-align:center;padding:0;position:relative">
            <div style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#333;transform:translateX(-50%)"></div>
            <div style="position:relative;width:14px;height:14px;border-radius:50%;background:{c};border:2px solid #111;margin:auto;z-index:1"></div>
          </td>
          <td style="padding:6px 0 6px 10px">
            <span style="font-size:.65rem;font-weight:700;color:{c};text-transform:uppercase;letter-spacing:.05em;margin-right:6px">{row['type']}</span>
            <span style="font-size:.75rem;color:#ccc">{summary}</span>
          </td>
        </tr>"""
    _components.html(f"""<div style="background:#111;padding:16px;border-radius:8px;font-family:system-ui,sans-serif">
  <table style="border-collapse:collapse;width:100%">{rows_html}</table>
</div>""", height=min(60 + n * 34, 600), scrolling=True)

    st.subheader("Events")
    type_options = ["all"] + sorted(EVENT_TYPES)
    channels = sorted({e.get("channel", "") for e in events if e.get("channel")})
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox("Type", type_options, key="tl_type")
    with col2:
        channel_filter = st.selectbox("Channel", ["all"] + channels, key="tl_channel")

    shown = [e for e in events
             if (type_filter == "all" or e["type"] == type_filter)
             and (channel_filter == "all" or e.get("channel") == channel_filter)]
    st.caption(f"{len(shown)} of {len(events)} events")

    for e in shown:
        with st.container(border=True):
            type_colors = {"decision": "#4C9BE8", "milestone": "#4CAF50", "blocker": "#E85454", "resolution": "#FFA726"}
            c = type_colors.get(e["type"], "#888")
            hdr = f"<span style='color:{c};font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:0.04em'>{e['type']}</span>"
            hdr += f" <span style='color:#666;font-size:0.75rem'>{e['date']}</span>"
            if e.get("channel"):
                hdr += f" <span style='color:#555;font-size:0.7rem'>#{e['channel']}</span>"
            st.markdown(hdr, unsafe_allow_html=True)
            st.markdown(e["summary"])


# --- main views --------------------------------------------------------------

def render_sidebar(messages):
    with st.sidebar:
        st.markdown("### Punch.io")
        projects = sorted({m.project for m in messages if m.project})
        # Messages synced before project tagging have project=None; give them their
        # own option so they can be isolated instead of only appearing under "All".
        has_unassigned = any(not m.project for m in messages)
        options = ["All projects"] + projects + (["Unassigned"] if has_unassigned else [])
        selected = None
        if len(options) > 1:
            choice = st.selectbox("Project", options)
            selected = None if choice == "All projects" else choice

        st.divider()
        st.metric("Messages", len(messages))
        st.metric("Index", "ready" if _index_ready() else "not built")

        if st.button("Rebuild search index", width="stretch",
                     help="Embed new messages into the vector index (live RAG build)."):
            with st.spinner("Embedding messages…"):
                try:
                    n = _rebuild_index()
                    st.success(f"Embedded {n} new message(s).")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

        if st.session_state.get("synced_channels"):
            st.divider()
            if st.button("Sync now", width="stretch",
                         help="Re-fetch messages from all synced channels."):
                token = st.session_state.get("discord_token") or os.environ.get("DISCORD_BOT_TOKEN")
                for ch in st.session_state.synced_channels:
                    connector = DiscordConnector(
                        channel=ch["name"],
                        channel_id=ch["id"],
                        bot_token=token,
                    )
                    try:
                        sync_source(connector, db_path=DB_PATH, faiss_dir=os.path.join(DATA_DIR, "faiss_db"))
                    except Exception as e:
                        st.error(f"#{ch['name']}: {e}")
                _rebuild_index()
                st.success("Sync complete.")
                st.rerun()

        st.divider()
        if st.button("Reconnect a different bot", width="stretch"):
            st.session_state["force_onboard"] = True
            st.session_state["onboarded"] = False
            st.rerun()

        return selected


def render_chat(messages):
    st.subheader("Ask about your project")

    if not _index_ready():
        st.warning("No search index yet. Click **Rebuild search index** in the sidebar "
                   "to enable Q&A.")
        return

    from generation import ask_question, retrieve_context
    from generation.llm import stream_answer

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    if q := st.chat_input("e.g. What's blocking the deployment?"):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching communications…"):
                history = [{"role": r, "text": c} for r, c in st.session_state.chat[:-1]]
                docs, _ = retrieve_context(q)
                answer = st.write_stream(stream_answer(
                    q, history=history,
                    api_key=st.session_state.get("groq_api_key")))
                st.markdown(answer)
                with st.expander(f"{len(docs)} source messages"):
                    for d in docs:
                        meta = d.metadata
                        st.markdown(
                            f"**{meta.get('author', '?')}** · "
                            f"`{meta.get('channel', '?')}` · "
                            f"{(meta.get('timestamp') or '')[:10]}"
                        )
                        st.caption(d.page_content)
        st.session_state.chat.append(("assistant", answer))


def render_messages(messages):
    st.subheader("Messages")
    if not messages:
        st.info("No messages in this project yet.")
        return

    channels = sorted({m.channel for m in messages if m.channel})
    picked = st.multiselect("Filter channels", channels, default=channels)
    shown = [m for m in messages if m.channel in picked]
    # Newest first. No truncation — showing all was the whole point (old code
    # dropped the oldest with messages[-50:], hiding un-scrolled history).
    shown = sorted(shown, key=lambda m: m.timestamp or "", reverse=True)
    st.caption(f"{len(shown)} of {len(messages)} messages")

    for m in shown:
        with st.container(border=True):
            cols = st.columns([1, 4, 2])
            cols[0].markdown(f"**{m.author}**", help=m.source if m.source else None)
            cols[1].markdown(f"<span style='color:#888;font-size:0.75rem'>#{m.channel}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='color:#555;font-size:0.7rem;float:right'>{(m.timestamp or '')[:16]}</span>", unsafe_allow_html=True)
            st.markdown(m.content)


def main():
    if st.session_state.get("force_onboard"):
        render_onboarding()
        return

    if not st.session_state.get("onboarded"):
        has_creds = (
            st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
        ) and (
            st.session_state.get("discord_token") or os.environ.get("DISCORD_BOT_TOKEN")
        )
        if _db_ready() and has_creds:
            store.init_db(DB_PATH)
            st.session_state.onboarded = True
            st.session_state.channels_setup = True
        else:
            render_onboarding()
            return

    store.init_db(DB_PATH)

    if not st.session_state.get("channels_setup"):
        render_channel_setup()
        return

    messages = _load_messages()
    if not messages:
        st.title("Punch.io")
        st.info("No messages yet. Go back and sync some channels.")
        if st.button("Back to channel setup"):
            st.session_state.channels_setup = False
            st.rerun()
        return

    selected = render_sidebar(messages)
    if selected == "Unassigned":
        messages = [m for m in messages if not m.project]
    elif selected is not None:
        messages = _load_messages(project=selected)

    TAB_ROUTES = {"chat": "ask", "timeline": "timeline", "messages": "messages"}
    tab_param = st.query_params.get("tab", "chat")
    tab = TAB_ROUTES.get(tab_param, "ask")
    nav_items = [("chat", "Ask"), ("timeline", "Timeline"), ("messages", "Messages")]
    cols = st.columns(len(nav_items))
    for i, (param, label) in enumerate(nav_items):
        with cols[i]:
            if st.button(label, use_container_width=True,
                         type="primary" if tab_param == param else "secondary"):
                st.query_params.tab = param
                st.rerun()

    if tab == "ask":
        render_chat(messages)
    elif tab == "messages":
        render_messages(messages)
    elif tab == "timeline":
        render_timeline_tab(messages)


if __name__ == "__main__":
    main()
