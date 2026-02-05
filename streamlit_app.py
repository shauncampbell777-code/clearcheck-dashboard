import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="ClearCheck Approval Dashboard", layout="wide")
st.title("ClearCheck Technologies — Approval Behavior Dashboard")
st.caption("Technician approval timing, fast approvals, and session patterns.")

COLOR_MAP = {"Arnold": "red", "Mendez": "blue", "Shawn": "green"}

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOUR_ORDER = list(range(24))


# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    # Works whether CSV is in same folder as app.py OR one level above
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "data_full.csv"),
        os.path.join(here, "..", "data_full.csv"),
    ]
    csv_path = None
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        raise FileNotFoundError("data_full.csv not found. Put it in the same folder as app.py (recommended).")

    df = pd.read_csv(csv_path)

    # Ensure datetime
    df["APPROVAL_DATE"] = pd.to_datetime(df["APPROVAL_DATE"], errors="coerce")
    df = df[df["APPROVAL_DATE"].notna()].copy()

    # Ensure TECHNICIAN exists
    if "TECHNICIAN" not in df.columns:
        raise ValueError("Missing TECHNICIAN column in CSV.")

    # Ensure DURATION_SEC exists; if not, compute per technician
    if "DURATION_SEC" not in df.columns:
        df = df.sort_values(["TECHNICIAN", "APPROVAL_DATE"]).copy()
        df["DURATION_SEC"] = df.groupby("TECHNICIAN")["APPROVAL_DATE"].diff().dt.total_seconds()

    # Keep valid durations only
    df = df[df["DURATION_SEC"].notna() & (df["DURATION_SEC"] >= 0)].copy()

    # Time features
    df["DATE"] = df["APPROVAL_DATE"].dt.date
    df["WEEKDAY"] = df["APPROVAL_DATE"].dt.day_name()
    df["HOUR_OF_DAY"] = df["APPROVAL_DATE"].dt.hour
    df["MONTH"] = df["APPROVAL_DATE"].dt.to_period("M").dt.to_timestamp()

    return df


df = load_data()

# Technician list (force consistent order)
tech_list = ["Arnold", "Mendez", "Shawn"]
tech_list_existing = [t for t in tech_list if t in df["TECHNICIAN"].unique()]
TECH_OPTIONS = ["All Tech"] + tech_list_existing


# ---------------- Sidebar Filters ----------------
with st.sidebar:
    st.header("Filters")
    tech = st.selectbox("Technician", TECH_OPTIONS)

    min_d = df["APPROVAL_DATE"].min().date()
    max_d = df["APPROVAL_DATE"].max().date()
    date_range = st.date_input("Date Range", (min_d, max_d))

    fast_threshold = st.slider("Fast Approval Threshold (sec)", 1, 60, 10)
    clip_sec = st.slider("Histogram Clip (sec)", 60, 1800, 600)
    bins_n = st.slider("Histogram Bins", 20, 120, 50)

    st.divider()
    st.subheader("Block Controls")
    block_gap_min = st.slider("New block if gap ≥ (minutes)", 5, 30, 10)
    min_cases_in_block = st.slider("Show blocks with at least (cases)", 1, 50, 1)


start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

df_range = df[(df["APPROVAL_DATE"] >= start) & (df["APPROVAL_DATE"] < end)].copy()

# Helper to get selected-tech subset
def get_selected_df():
    if tech == "All Tech":
        return df_range.copy()
    return df_range[df_range["TECHNICIAN"] == tech].copy()


d = get_selected_df().sort_values("APPROVAL_DATE").copy()

# Add quick flags
if len(d) > 0:
    d["DURATION_MIN"] = d["DURATION_SEC"] / 60
    d["FAST"] = d["DURATION_SEC"] < fast_threshold
    d["SAME_SECOND"] = d["DURATION_SEC"] == 0
    d["SAME_MINUTE"] = d["DURATION_SEC"] < 60


# ---------------- Top KPI Strip ----------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Rows in Filter", f"{len(d):,}")
k2.metric("Date Min", str(df_range["APPROVAL_DATE"].min().date()) if len(df_range) else "—")
k3.metric("Date Max", str(df_range["APPROVAL_DATE"].max().date()) if len(df_range) else "—")

if len(d) > 0 and tech != "All Tech":
    k4.metric("Median Duration (sec)", f"{d['DURATION_SEC'].median():.2f}")
    k5.metric("% Fast", f"{(d['FAST'].mean()*100):.2f}%")
else:
    k4.metric("Median Duration (sec)", "—")
    k5.metric("% Fast", "—")

st.divider()


# ---------------- Tabs ----------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview (All Tech)",
    "Histogram",
    "Fast Approvals",
    "Blocks / Sessions",
    "Time Patterns",
    "High-Risk",
    "Policy Change (Arnold)"
])


# =========================
# TAB 0: OVERVIEW (ALL TECH)
# =========================
with tab0:
    st.subheader("All Technicians Overview")
    st.caption("This tab is only meaningful when you choose 'All Tech' in the sidebar.")

    if tech != "All Tech":
        st.info("Switch Technician to **All Tech** to view cross-technician comparisons.")
    elif len(df_range) == 0:
        st.warning("No data available for the selected date range.")
    else:
        # 1) Total approvals by tech
        st.markdown("### Total Approvals by Technician")
        counts = df_range["TECHNICIAN"].value_counts().reindex(tech_list_existing, fill_value=0)

        fig = plt.figure(figsize=(7, 4))
        plt.bar(
            counts.index,
            counts.values,
            color=[COLOR_MAP.get(t, "gray") for t in counts.index],
            edgecolor="black"
        )
        plt.ylabel("Approvals")
        plt.xlabel("Technician")
        plt.title("Approved Cases Count")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

        # 2) Daily approvals over time (lines by tech)
        st.markdown("### Daily Approvals Over Time")
        daily = df_range.groupby(["DATE", "TECHNICIAN"]).size().reset_index(name="approvals")

        fig = plt.figure(figsize=(10, 4))
        for t in tech_list_existing:
            sub = daily[daily["TECHNICIAN"] == t]
            plt.plot(sub["DATE"], sub["approvals"], label=t, color=COLOR_MAP.get(t, "gray"), linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Approvals")
        plt.title("Daily Approvals (All Tech)")
        plt.grid(alpha=0.25)
        plt.legend(title="Technician")
        plt.tight_layout()
        st.pyplot(fig)

        # 3) Daily fast % over time
        st.markdown(f"### Daily Fast Approval Rate (< {fast_threshold}s)")
        tmp = df_range.copy()
        tmp["FAST"] = tmp["DURATION_SEC"] < fast_threshold
        daily_fast = tmp.groupby(["DATE", "TECHNICIAN"]).agg(pct_fast=("FAST", "mean")).reset_index()
        daily_fast["pct_fast"] *= 100

        fig = plt.figure(figsize=(10, 4))
        for t in tech_list_existing:
            sub = daily_fast[daily_fast["TECHNICIAN"] == t]
            plt.plot(sub["DATE"], sub["pct_fast"], label=t, color=COLOR_MAP.get(t, "gray"), linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("% Fast")
        plt.ylim(0, 100)
        plt.title("Daily % Fast Approvals (All Tech)")
        plt.grid(alpha=0.25)
        plt.legend(title="Technician")
        plt.tight_layout()
        st.pyplot(fig)

        # 4) Boxplot clipped
        st.markdown("### Boxplot of Approval Durations (Clipped)")
        clip_for_box = 600
        box_df = df_range.copy()
        box_df["DUR_CLIPPED"] = box_df["DURATION_SEC"].clip(upper=clip_for_box)

        fig = plt.figure(figsize=(7, 4))
        data_to_plot = [box_df[box_df["TECHNICIAN"] == t]["DUR_CLIPPED"] for t in tech_list_existing]
        plt.boxplot(data_to_plot, labels=tech_list_existing, showfliers=False)
        plt.title("Duration Distribution (sec, clipped at 600)")
        plt.ylabel("Seconds")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

        # 5) Bubble blocks: all tech (compare)
        st.markdown("### Blocks Bubble Chart (All Tech)")
        st.caption("Each bubble is a session (block). Bigger bubbles = more approvals in that session.")

        all_blocks = []
        for t in tech_list_existing:
            dt = df_range[df_range["TECHNICIAN"] == t].sort_values("APPROVAL_DATE").copy()
            dt["GAP_MIN"] = dt["DURATION_SEC"] / 60
            dt["NEW_BLOCK"] = dt["GAP_MIN"] >= block_gap_min
            dt["BLOCK_ID"] = dt["NEW_BLOCK"].cumsum()

            blk = dt.groupby("BLOCK_ID").agg(
                cases_in_block=("DURATION_SEC", "count"),
                avg_gap=("DURATION_SEC", "mean"),
                start=("APPROVAL_DATE", "min"),
                end=("APPROVAL_DATE", "max"),
            ).reset_index()
            blk["TECHNICIAN"] = t
            all_blocks.append(blk)

        block_summary = pd.concat(all_blocks, ignore_index=True)
        block_summary = block_summary[block_summary["cases_in_block"] >= min_cases_in_block].copy()

        if len(block_summary) == 0:
            st.warning("No blocks match your block filters.")
        else:
            # sample if too large
            sampled = block_summary.sample(frac=0.15, random_state=42) if len(block_summary) > 2500 else block_summary

            fig = plt.figure(figsize=(10, 6))
            for t in tech_list_existing:
                sub = sampled[sampled["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue
                sizes = (sub["cases_in_block"] / sub["cases_in_block"].max()) * 900 + 40
                plt.scatter(
                    sub["cases_in_block"], sub["avg_gap"],
                    s=sizes,
                    alpha=0.6,
                    color=COLOR_MAP.get(t, "gray"),
                    edgecolors="black",
                    linewidth=0.5,
                    label=t
                )
            plt.xlabel("Cases in Block")
            plt.ylabel("Average Gap (sec)")
            plt.title("Cases per Block vs Avg Gap (All Tech)")
            plt.grid(alpha=0.25)
            plt.legend(title="Technician")
            plt.tight_layout()
            st.pyplot(fig)

        st.download_button(
            "Download All-Tech block summary (CSV)",
            block_summary.to_csv(index=False),
            "alltech_block_summary.csv",
            "text/csv"
        )


# =================
# TAB 1: HISTOGRAM
# =================
with tab1:
    st.subheader("Approval Duration Histogram")

    if tech == "All Tech":
        st.info("Histogram is shown per technician in **All Tech** mode (three separate plots).")
        show_techs = tech_list_existing
    else:
        show_techs = [tech]

    if len(df_range) == 0:
        st.warning("No data in selected date range.")
    else:
        for t in show_techs:
            dt = df_range[df_range["TECHNICIAN"] == t].copy()
            if len(dt) == 0:
                continue

            dt["DUR_CLIPPED"] = dt["DURATION_SEC"].clip(upper=clip_sec)
            bins = np.linspace(0, clip_sec, bins_n)

            fig = plt.figure(figsize=(10, 4))
            counts, bin_edges, _ = plt.hist(
                dt["DUR_CLIPPED"],
                bins=bins,
                alpha=0.85,
                edgecolor="black",
                color=COLOR_MAP.get(t, "gray")
            )
            plt.yscale("log")  # keep your log scale

            # Peak label
            if len(counts) > 0:
                max_count = counts.max()
                max_index = counts.argmax()
                x_peak = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
                plt.annotate(
                    f"Peak: {int(max_count)} approvals",
                    xy=(x_peak, max_count),
                    xytext=(x_peak, max_count * 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5),
                    ha="center",
                    fontsize=10,
                    weight="bold"
                )

            plt.xlabel("Approval Duration (seconds, clipped)")
            plt.ylabel("Approvals (log scale)")
            plt.title(f"{t} — Approval Duration Histogram")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)

        st.caption("Why log scale? Because most approvals are very fast, but a small number are much slower. Log scale makes both visible.")


# ====================
# TAB 2: FAST APPROVAL
# ====================
with tab2:
    st.subheader("Fast Approval Rate")
    st.caption("Higher percentages at low thresholds may indicate rushed approvals.")

    thresholds = [2, 5, 10, 30, 60]

    if len(df_range) == 0:
        st.warning("No data in selected date range.")
    else:
        if tech == "All Tech":
            st.info("All Tech view shows three separate fast-rate charts (one per technician).")
            show_techs = tech_list_existing
        else:
            show_techs = [tech]

        for t in show_techs:
            dt = df_range[df_range["TECHNICIAN"] == t]
            if len(dt) == 0:
                continue
            rates = [(dt["DURATION_SEC"] < thr).mean() * 100 for thr in thresholds]

            fig = plt.figure(figsize=(6, 4))
            plt.plot(thresholds, rates, marker="o", linewidth=3, color=COLOR_MAP.get(t, "gray"))
            plt.xlabel("Threshold (seconds)")
            plt.ylabel("% Fast Approvals")
            plt.title(f"{t} — Fast Approval Rate")
            plt.ylim(0, 100)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

        # Quick insight text
        if tech != "All Tech" and len(d) > 0:
            pct10 = (d["DURATION_SEC"] < 10).mean() * 100
            st.success(f"Insight: **{tech}** has **{pct10:.2f}%** approvals under **10 seconds** in this date range.")


# =====================
# TAB 3: BLOCKS/SESSIONS
# =====================
with tab3:
    st.subheader("Block / Session Analysis")
    st.caption(f"A new block starts when the gap between approvals is ≥ {block_gap_min} minutes.")

    if tech == "All Tech":
        st.info("This tab is designed for **one technician at a time**. Choose a technician in the sidebar.")
    elif len(d) == 0:
        st.warning("No approvals in this filter window.")
    else:
        dt = d.sort_values("APPROVAL_DATE").copy()
        dt["GAP_MIN"] = dt["DURATION_SEC"] / 60
        dt["NEW_BLOCK"] = dt["GAP_MIN"] >= block_gap_min
        dt["BLOCK_ID"] = dt["NEW_BLOCK"].cumsum()

        blocks = dt.groupby("BLOCK_ID").agg(
            cases=("DURATION_SEC", "count"),
            avg_gap=("DURATION_SEC", "mean"),
            median_gap=("DURATION_SEC", "median"),
            start=("APPROVAL_DATE", "min"),
            end=("APPROVAL_DATE", "max"),
            pct_fast_10s=("DURATION_SEC", lambda x: (x < 10).mean() * 100),
        ).reset_index()

        blocks = blocks[blocks["cases"] >= min_cases_in_block].copy()

        # KPI for sessions
        a, b, c = st.columns(3)
        a.metric("Blocks in Range", f"{len(blocks):,}")
        b.metric("Median Cases/Block", f"{blocks['cases'].median():.0f}" if len(blocks) else "—")
        c.metric("Max Cases in a Block", f"{blocks['cases'].max():.0f}" if len(blocks) else "—")

        st.download_button(
            "Download block table (CSV)",
            blocks.to_csv(index=False),
            f"{tech}_block_summary.csv",
            "text/csv"
        )

        st.markdown("### Bubble Chart: Cases per Block vs Avg Gap")
        fig = plt.figure(figsize=(10, 6))
        sizes = (blocks["cases"] / blocks["cases"].max()) * 800 + 50 if len(blocks) else []

        plt.scatter(
            blocks["cases"], blocks["avg_gap"],
            s=sizes,
            alpha=0.65,
            color=COLOR_MAP.get(tech, "gray"),
            edgecolors="black",
            linewidth=0.5
        )
        plt.xlabel("Number of Cases in Block")
        plt.ylabel("Average Gap Between Approvals (sec)")
        plt.title(f"{tech} — Session Bubble Chart")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Largest Blocks (Top 10)")
        st.dataframe(
            blocks.sort_values("cases", ascending=False).head(10)[
                ["BLOCK_ID","cases","avg_gap","median_gap","pct_fast_10s","start","end"]
            ],
            use_container_width=True
        )


# =================
# TAB 4: TIME PATTERNS
# =================
with tab4:
    st.subheader("Time Patterns")

    if len(d) == 0:
        st.warning("No data available for selected filters.")
    else:
        if tech == "All Tech":
            st.info("Pick a technician to view weekday/hour patterns. (All Tech comparisons are in Overview tab.)")
        else:
            c1, c2 = st.columns(2)

            # Weekday bar
            with c1:
                weekday_counts = d["WEEKDAY"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
                fig = plt.figure(figsize=(7, 4))
                plt.bar(weekday_counts.index, weekday_counts.values, color=COLOR_MAP.get(tech, "gray"), edgecolor="black")
                plt.title(f"{tech} — Approvals by Weekday")
                plt.xlabel("Weekday")
                plt.ylabel("Approvals")
                plt.xticks(rotation=30)
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            # Hour bar
            with c2:
                hour_counts = d["HOUR_OF_DAY"].value_counts().reindex(HOUR_ORDER, fill_value=0)
                fig = plt.figure(figsize=(7, 4))
                plt.bar(hour_counts.index, hour_counts.values, color=COLOR_MAP.get(tech, "gray"), edgecolor="black")
                plt.title(f"{tech} — Approvals by Hour")
                plt.xlabel("Hour (0–23)")
                plt.ylabel("Approvals")
                plt.xticks(range(0, 24, 2))
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("### Heatmap: Weekday × Hour (Approval Concentration)")
            heat = (
                d.pivot_table(index="WEEKDAY", columns="HOUR_OF_DAY", values="DURATION_SEC", aggfunc="count", fill_value=0)
                .reindex(index=WEEKDAY_ORDER, columns=HOUR_ORDER, fill_value=0)
            )

            fig = plt.figure(figsize=(12, 4))
            plt.imshow(heat.values, aspect="auto")
            plt.yticks(range(len(WEEKDAY_ORDER)), WEEKDAY_ORDER)
            plt.xticks(range(0, 24, 2), range(0, 24, 2))
            plt.xlabel("Hour")
            plt.ylabel("Weekday")
            plt.title(f"{tech} — Activity Heatmap (Counts)")
            plt.colorbar(label="Approvals")
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Show heatmap table"):
                st.dataframe(heat, use_container_width=True)


# =================
# TAB 5: HIGH RISK
# =================
with tab5:
    st.subheader("High-Risk Fast Approvals (Worst Cases)")
    st.caption("These are the lowest approval gaps (fastest consecutive approvals).")

    if tech == "All Tech":
        st.info("Pick a technician to view worst-case approvals.")
    elif len(d) == 0:
        st.warning("No data available.")
    else:
        worst = d.sort_values("DURATION_SEC").head(50)[["APPROVAL_DATE","CASE_NUMBER","DURATION_SEC"]].copy()
        st.dataframe(worst, use_container_width=True)

        st.download_button(
            "Download worst cases (CSV)",
            worst.to_csv(index=False),
            f"{tech}_worst_cases.csv",
            "text/csv"
        )


# ==========================
# TAB 6: POLICY CHANGE (ARNOLD)
# ==========================
with tab6:
    st.subheader("Policy Change Behavior — Arnold")
    st.caption("Cutoff date: June 1, 2020 (payout reduced from $50 to $17).")

    cutoff = pd.Timestamp("2020-06-01")
    arn = df_range[df_range["TECHNICIAN"] == "Arnold"].copy()

    if len(arn) == 0:
        st.warning("No Arnold data in selected date range.")
    else:
        before = arn[arn["APPROVAL_DATE"] < cutoff]
        after = arn[arn["APPROVAL_DATE"] >= cutoff]

        st.markdown("### Monthly % Fast (<10s) Trend")
        arn["FAST_10S"] = arn["DURATION_SEC"] < 10
        monthly = arn.groupby("MONTH").agg(pct_fast=("FAST_10S", "mean")).reset_index()
        monthly["pct_fast"] *= 100

        fig = plt.figure(figsize=(10, 4))
        plt.plot(monthly["MONTH"], monthly["pct_fast"], marker="o", linewidth=3, color="red")
        plt.axvline(cutoff, linestyle="--", linewidth=2, color="black")
        plt.xlabel("Month")
        plt.ylabel("% Approvals < 10 sec")
        plt.title("Arnold — Fast Approval Trend Over Time")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Before vs After Summary")
        summary = pd.DataFrame({
            "Period": ["Before Policy", "After Policy"],
            "Total Approvals": [len(before), len(after)],
            "Mean Duration (sec)": [round(before["DURATION_SEC"].mean(), 2), round(after["DURATION_SEC"].mean(), 2)],
            "Median Duration (sec)": [round(before["DURATION_SEC"].median(), 2), round(after["DURATION_SEC"].median(), 2)],
            "% Fast (<10s)": [
                round((before["DURATION_SEC"] < 10).mean() * 100, 2),
                round((after["DURATION_SEC"] < 10).mean() * 100, 2)
            ]
        })
        st.dataframe(summary, use_container_width=True)

        # BONUS: Find big blocks >=200 approvals (date/time)
        st.markdown("### Bonus: Find Arnold blocks with ≥ 200 approvals")
        dt = arn.sort_values("APPROVAL_DATE").copy()
        dt["GAP_MIN"] = dt["DURATION_SEC"] / 60
        dt["NEW_BLOCK"] = dt["GAP_MIN"] >= block_gap_min
        dt["BLOCK_ID"] = dt["NEW_BLOCK"].cumsum()

        big_blocks = dt.groupby("BLOCK_ID").agg(
            cases=("DURATION_SEC", "count"),
            start=("APPROVAL_DATE", "min"),
            end=("APPROVAL_DATE", "max")
        ).reset_index()

        big_blocks = big_blocks[big_blocks["cases"] >= 200].sort_values("cases", ascending=False)

        if len(big_blocks) == 0:
            st.info("No Arnold block with ≥ 200 approvals found under current block gap rule.")
        else:
            st.dataframe(big_blocks, use_container_width=True)
