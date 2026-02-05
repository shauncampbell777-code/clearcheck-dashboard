import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="ClearCheck Approval Dashboard", layout="wide")

TECH_ORDER = ["Arnold", "Mendez", "Shawn"]
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOUR_ORDER = list(range(24))

COLOR_MAP = {"Arnold": "red", "Mendez": "blue", "Shawn": "green", "All Tech": "gray"}

st.title("ClearCheck Technologies — Approval Behavior Dashboard")
st.caption("Technician approval timing, fast approvals, and session patterns.")

# ---------------------------
# LOAD DATA (ONCE)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_full.csv")

    df["APPROVAL_DATE"] = pd.to_datetime(df["APPROVAL_DATE"], errors="coerce")
    df = df[df["APPROVAL_DATE"].notna()].copy()

    # Make sure duration is numeric and valid
    df["DURATION_SEC"] = pd.to_numeric(df["DURATION_SEC"], errors="coerce")
    df = df[df["DURATION_SEC"].notna() & (df["DURATION_SEC"] >= 0)].copy()

    # Add time columns
    df["DATE"] = df["APPROVAL_DATE"].dt.date
    df["WEEKDAY"] = df["APPROVAL_DATE"].dt.day_name()
    df["HOUR_OF_DAY"] = df["APPROVAL_DATE"].dt.hour
    df["MONTH"] = df["APPROVAL_DATE"].dt.to_period("M").dt.to_timestamp()

    return df

df = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
with st.sidebar:
    st.header("Filters")

    tech_options = ["All Tech"] + TECH_ORDER
    tech = st.selectbox("Technician", tech_options, index=0)

    min_d = df["APPROVAL_DATE"].min().date()
    max_d = df["APPROVAL_DATE"].max().date()
    date_range = st.date_input("Date Range", (min_d, max_d))

    fast_threshold = st.slider("Fast Approval Threshold (sec)", 1, 60, 10)
    clip_sec = st.slider("Histogram Clip (sec)", 60, 1800, 600)
    bins_n = st.slider("Histogram Bins", 20, 120, 50)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

# ---------------------------
# FILTERED DATA (d)
# ---------------------------
base = df[(df["APPROVAL_DATE"] >= start) & (df["APPROVAL_DATE"] < end)].copy()

if tech == "All Tech":
    d = base.copy()
else:
    d = base[base["TECHNICIAN"] == tech].copy()

# Derived flags for filtered dataset
d["FAST"] = d["DURATION_SEC"] < fast_threshold
d["SAME_SECOND"] = d["DURATION_SEC"] == 0
d["SAME_MINUTE"] = d["DURATION_SEC"] < 60
d["DURATION_MIN"] = d["DURATION_SEC"] / 60

# ---------------------------
# KPIs
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Approvals", f"{len(d):,}")
c2.metric("Median (sec)", round(d["DURATION_SEC"].median(), 2) if len(d) else "N/A")
c3.metric("% Fast", round(d["FAST"].mean() * 100, 2) if len(d) else "N/A")
c4.metric("% Same Minute", round(d["SAME_MINUTE"].mean() * 100, 2) if len(d) else "N/A")

st.divider()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Histogram",
    "Fast Approvals",
    "Blocks / Sessions",
    "Time Patterns",
    "High-Risk Fast Approvals",
    "Policy Change (Arnold)"
])

# ---------------------------
# TAB 1: Histogram
# ---------------------------
with tab1:
    st.subheader("Approval Duration Histogram")

    if len(d) == 0:
        st.warning("No data for the selected filters.")
    else:
        d_plot = d.copy()
        d_plot["DUR_CLIPPED"] = d_plot["DURATION_SEC"].clip(upper=clip_sec)
        bins = np.linspace(0, clip_sec, bins_n)

        fig = plt.figure(figsize=(10, 4))
        counts, bin_edges, _ = plt.hist(
            d_plot["DUR_CLIPPED"],
            bins=bins,
            alpha=0.85,
            edgecolor="black",
            color=COLOR_MAP.get(tech, "gray")
        )
        plt.yscale("log")

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
        plt.ylabel("Number of Approvals (log scale)")
        plt.title(f"{tech} — Approval Duration Histogram")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------
# TAB 2: Fast approvals
# ---------------------------
with tab2:
    st.subheader("Fast Approval Rate")

    if len(d) == 0:
        st.warning("No data for the selected filters.")
    else:
        thresholds = [2, 5, 10, 30, 60]

        if tech == "All Tech":
            fig = plt.figure(figsize=(7, 4))
            for t in TECH_ORDER:
                dt = base[base["TECHNICIAN"] == t].copy()
                rates = [(dt["DURATION_SEC"] < x).mean() * 100 for x in thresholds]
                plt.plot(thresholds, rates, marker="o", linewidth=2, label=t, color=COLOR_MAP[t])
            plt.legend(title="Technician")
            plt.title("Fast Approval Rate — All Tech")
        else:
            rates = [(d["DURATION_SEC"] < x).mean() * 100 for x in thresholds]
            fig = plt.figure(figsize=(7, 4))
            plt.plot(thresholds, rates, marker="o", linewidth=3, color=COLOR_MAP.get(tech, "gray"))
            plt.title(f"{tech} — Fast Approval Rate")

        plt.xlabel("Threshold (seconds)")
        plt.ylabel("% Fast Approvals")
        plt.ylim(0, 100)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------
# TAB 3: Blocks / Sessions
# ---------------------------
with tab3:
    st.subheader("Blocks / Sessions")

    if len(d) == 0:
        st.warning("No data for the selected filters.")
    else:
        st.caption("Blocks are approximated using the existing DURATION_SEC as the gap proxy (>= 10 minutes = new block).")

        if tech == "All Tech":
            # Build blocks per tech and combine into one bubble chart (the screenshot style)
            all_blocks = []
            for t in TECH_ORDER:
                dt = base[base["TECHNICIAN"] == t].sort_values("APPROVAL_DATE").copy()
                dt["NEW_BLOCK"] = (dt["DURATION_SEC"] / 60) >= 10
                dt["BLOCK_ID"] = dt["NEW_BLOCK"].cumsum()

                blk = dt.groupby("BLOCK_ID").agg(
                    cases_in_block=("DURATION_SEC", "count"),
                    avg_gap_sec=("DURATION_SEC", "mean")
                ).reset_index()

                blk["TECHNICIAN"] = t
                all_blocks.append(blk)

            block_summary = pd.concat(all_blocks, ignore_index=True)

            fig = plt.figure(figsize=(12, 6))
            for t in TECH_ORDER:
                sub = block_summary[block_summary["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue
                sizes = (sub["cases_in_block"] / sub["cases_in_block"].max()) * 900 + 40
                plt.scatter(
                    sub["cases_in_block"],
                    sub["avg_gap_sec"],
                    s=sizes,
                    alpha=0.6,
                    color=COLOR_MAP[t],
                    edgecolors="black",
                    linewidth=0.5,
                    label=t
                )

            plt.title("Approval per Block vs. Average Gap (All Tech)")
            plt.xlabel("Number of Cases in Block")
            plt.ylabel("Average Gap (seconds) within block")
            plt.grid(alpha=0.25)
            plt.legend(title="Technician")
            plt.tight_layout()
            st.pyplot(fig)

        else:
            # Single-tech block chart
            dt = d.sort_values("APPROVAL_DATE").copy()
            dt["NEW_BLOCK"] = (dt["DURATION_SEC"] / 60) >= 10
            dt["BLOCK_ID"] = dt["NEW_BLOCK"].cumsum()

            block = dt.groupby("BLOCK_ID").agg(
                cases=("DURATION_SEC", "count"),
                avg_gap=("DURATION_SEC", "mean")
            ).reset_index()

            fig = plt.figure(figsize=(10, 6))
            sizes = (block["cases"] / block["cases"].max()) * 900 + 40

            plt.scatter(
                block["cases"],
                block["avg_gap"],
                s=sizes,
                alpha=0.65,
                color=COLOR_MAP.get(tech, "gray"),
                edgecolors="black",
                linewidth=0.5
            )

            plt.title(f"{tech} — Cases per Block vs Avg Gap")
            plt.xlabel("Number of Cases in Block")
            plt.ylabel("Average Gap (seconds)")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

# ---------------------------
# TAB 4: Time patterns (weekday + hour + heatmap)
# ---------------------------
with tab4:
    st.subheader("Time Patterns")

    if len(d) == 0:
        st.warning("No data for the selected filters.")
    else:
        if tech == "All Tech":
            st.caption("Showing patterns for each technician (All Tech).")

            col1, col2 = st.columns(2)

            # Weekday chart
            with col1:
                fig = plt.figure(figsize=(7, 4))
                for t in TECH_ORDER:
                    sub = base[base["TECHNICIAN"] == t]
                    wk = sub["WEEKDAY"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
                    plt.plot(WEEKDAY_ORDER, wk.values, marker="o", linewidth=2, label=t, color=COLOR_MAP[t])
                plt.xticks(rotation=25)
                plt.title("Approvals by Weekday — All Tech")
                plt.xlabel("Weekday")
                plt.ylabel("Approvals")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)

            # Hour chart
            with col2:
                fig = plt.figure(figsize=(7, 4))
                for t in TECH_ORDER:
                    sub = base[base["TECHNICIAN"] == t]
                    hr = sub["HOUR_OF_DAY"].value_counts().reindex(HOUR_ORDER, fill_value=0)
                    plt.plot(HOUR_ORDER, hr.values, marker="o", linewidth=2, label=t, color=COLOR_MAP[t])
                plt.xticks(range(0, 24, 2))
                plt.title("Approvals by Hour — All Tech")
                plt.xlabel("Hour (0–23)")
                plt.ylabel("Approvals")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)

        else:
            col1, col2 = st.columns(2)

            with col1:
                wk = d["WEEKDAY"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
                fig = plt.figure(figsize=(7, 4))
                plt.bar(wk.index, wk.values, color=COLOR_MAP.get(tech, "gray"), edgecolor="black")
                plt.xticks(rotation=25)
                plt.title(f"{tech} — Approvals by Weekday")
                plt.xlabel("Weekday")
                plt.ylabel("Approvals")
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                hr = d["HOUR_OF_DAY"].value_counts().reindex(HOUR_ORDER, fill_value=0)
                fig = plt.figure(figsize=(7, 4))
                plt.bar(hr.index, hr.values, color=COLOR_MAP.get(tech, "gray"), edgecolor="black")
                plt.xticks(range(0, 24, 2))
                plt.title(f"{tech} — Approvals by Hour")
                plt.xlabel("Hour (0–23)")
                plt.ylabel("Approvals")
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            st.divider()

            st.subheader("Weekday × Hour Heatmap")
            heat = (
                d.pivot_table(index="WEEKDAY", columns="HOUR_OF_DAY", values="CASE_NUMBER",
                              aggfunc="count", fill_value=0)
                .reindex(index=WEEKDAY_ORDER, columns=HOUR_ORDER, fill_value=0)
            )

            fig = plt.figure(figsize=(12, 4))
            plt.imshow(heat.values, aspect="auto")
            plt.yticks(range(len(WEEKDAY_ORDER)), WEEKDAY_ORDER)
            plt.xticks(range(0, 24, 2), range(0, 24, 2))
            plt.xlabel("Hour of Day")
            plt.ylabel("Weekday")
            plt.title(f"{tech} — Approval Concentration (Counts)")
            plt.colorbar(label="Approvals")
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Show heatmap table"):
                st.dataframe(heat, use_container_width=True)

# ---------------------------
# TAB 5: High risk (fastest)
# ---------------------------
with tab5:
    st.subheader("High-Risk Fast Approvals (Fastest Records)")

    if len(d) == 0:
        st.warning("No data for the selected filters.")
    else:
        worst = d.sort_values("DURATION_SEC").head(50)
        cols = [c for c in ["APPROVAL_DATE", "TECHNICIAN", "CASE_NUMBER", "DURATION_SEC"] if c in worst.columns]
        st.dataframe(worst[cols], use_container_width=True)

# ---------------------------
# TAB 6: Policy change (Arnold)
# ---------------------------
with tab6:
    st.subheader("Policy Change Behavior")
    st.caption("Cutoff date: June 1, 2020 (payout reduced from $50 to $17)")
    cutoff = pd.Timestamp("2020-06-01")

    arn = df[df["TECHNICIAN"] == "Arnold"].copy()
    before = arn[arn["APPROVAL_DATE"] < cutoff]
    after  = arn[arn["APPROVAL_DATE"] >= cutoff]

    # Monthly fast trend (<10 sec)
    arn["FAST_10S"] = arn["DURATION_SEC"] < 10
    monthly = arn.groupby("MONTH").agg(pct_fast=("FAST_10S", "mean")).reset_index()

    fig = plt.figure(figsize=(10, 4))
    plt.plot(monthly["MONTH"], monthly["pct_fast"] * 100, marker="o", linewidth=3, color="red")
    plt.axvline(cutoff, linestyle="--", linewidth=2, color="black")
    plt.xlabel("Month")
    plt.ylabel("% Approvals < 10 sec")
    plt.title("Arnold Fast Approval Trend Over Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)

    # Summary table
    summary = pd.DataFrame({
        "Period": ["Before Policy", "After Policy"],
        "Total Approvals": [len(before), len(after)],
        "Mean Duration (sec)": [round(before["DURATION_SEC"].mean(), 2), round(after["DURATION_SEC"].mean(), 2)],
        "Median Duration (sec)": [round(before["DURATION_SEC"].median(), 2), round(after["DURATION_SEC"].median(), 2)],
        "% Fast (<10s)": [round((before["DURATION_SEC"] < 10).mean() * 100, 2),
                         round((after["DURATION_SEC"] < 10).mean() * 100, 2)]
    })
    st.dataframe(summary, use_container_width=True)
