import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page ----------------
st.set_page_config(page_title="ClearCheck Approval Dashboard", layout="wide")
st.title("ClearCheck Technologies — Approval Behavior Dashboard")
st.caption("Technician approval timing, fast approvals, and session patterns.")

COLOR_MAP = {"Arnold": "red", "Mendez": "blue", "Shawn": "green"}
TECH_ORDER = ["Arnold", "Mendez", "Shawn"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------- Load data (ONCE) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_full.csv")
    df["APPROVAL_DATE"] = pd.to_datetime(df["APPROVAL_DATE"], errors="coerce")
    df = df[df["APPROVAL_DATE"].notna()].copy()

    # Keep only valid durations already in your CSV
    df = df[df["DURATION_SEC"].notna() & (df["DURATION_SEC"] >= 0)].copy()

    # Time features
    df["DATE"] = df["APPROVAL_DATE"].dt.date
    df["WEEKDAY"] = df["APPROVAL_DATE"].dt.day_name()
    df["HOUR_OF_DAY"] = df["APPROVAL_DATE"].dt.hour

    return df

df = load_data()

# ---------------- Sidebar filters ----------------
with st.sidebar:
    tech = st.selectbox("Technician", ["All Tech"] + TECH_ORDER)

    min_d = df["APPROVAL_DATE"].min().date()
    max_d = df["APPROVAL_DATE"].max().date()
    date_range = st.date_input("Date Range", (min_d, max_d))

    fast_threshold = st.slider("Fast Approval Threshold (sec)", 1, 60, 10)
    clip_sec = st.slider("Histogram Clip (sec)", 60, 1800, 600)
    bins_n = st.slider("Histogram Bins", 20, 120, 50)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

# Date-range dataset for all tech
d_all = df[(df["APPROVAL_DATE"] >= start) & (df["APPROVAL_DATE"] < end)].copy()

# Technician dataset (for tabs that need selected tech)
if tech == "All Tech":
    d = d_all.copy()
else:
    d = d_all[d_all["TECHNICIAN"] == tech].copy()

# Derived flags
d["DURATION_MIN"] = d["DURATION_SEC"] / 60
d["FAST"] = d["DURATION_SEC"] < fast_threshold
d["SAME_SECOND"] = d["DURATION_SEC"] == 0
d["SAME_MINUTE"] = d["DURATION_SEC"] < 60

# ---------------- KPI cards ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Approvals", f"{len(d):,}")
c2.metric("Median (sec)", round(d["DURATION_SEC"].median(), 2) if len(d) else "N/A")
c3.metric("% Fast", round(d["FAST"].mean() * 100, 2) if len(d) else "N/A")
c4.metric("% Same Minute", round(d["SAME_MINUTE"].mean() * 100, 2) if len(d) else "N/A")

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Histogram",
    "Fast Approvals",
    "Blocks / Sessions",
    "Time Patterns",
    "High-Risk Fast Approvals",
    "Policy Change"
])

# ---------------- TAB 1: Histogram ----------------
with tab1:
    st.subheader("Approval Duration Histogram")
    st.caption("Log y-scale helps reveal the long tail (rare but large durations).")

    if len(d) == 0:
        st.warning("No data available for the selected filters.")
    else:
        d_plot = d.copy()
        d_plot["DUR_CLIPPED"] = d_plot["DURATION_SEC"].clip(upper=clip_sec)
        bins = np.linspace(0, clip_sec, bins_n)

        if tech == "All Tech":
            for t in TECH_ORDER:
                sub = d_plot[d_plot["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue

                fig = plt.figure(figsize=(10, 4))
                counts, bin_edges, _ = plt.hist(
                    sub["DUR_CLIPPED"],
                    bins=bins,
                    alpha=0.85,
                    edgecolor="black",
                    color=COLOR_MAP.get(t, "gray")
                )
                plt.yscale("log")

                if len(counts) > 0:
                    max_count = counts.max()
                    max_index = counts.argmax()
                    x_peak = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
                    plt.annotate(
                        f"Peak: {int(max_count)}",
                        xy=(x_peak, max_count),
                        xytext=(x_peak, max_count * 2),
                        arrowprops=dict(arrowstyle="->", lw=1.5),
                        ha="center",
                        fontsize=10,
                        weight="bold"
                    )

                plt.xlabel("Approval Duration (seconds, clipped)")
                plt.ylabel("Number of Approvals (log scale)")
                plt.title(f"{t} — Approval Duration Histogram")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig)
        else:
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
                    f"Peak: {int(max_count)}",
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

# ---------------- TAB 2: Fast approvals ----------------
with tab2:
    st.subheader("Fast Approval Rate")
    st.caption("Percent of approvals faster than each threshold (lower seconds + higher % can indicate rushed approvals).")

    thresholds = [2, 5, 10, 30, 60]

    if len(d) == 0:
        st.warning("No data available.")
    else:
        if tech == "All Tech":
            fig = plt.figure(figsize=(8, 4))
            for t in TECH_ORDER:
                sub = d[d["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue
                rates = [(sub["DURATION_SEC"] < x).mean() * 100 for x in thresholds]
                plt.plot(thresholds, rates, marker="o", linewidth=3, color=COLOR_MAP.get(t, "gray"), label=t)

            plt.xlabel("Threshold (seconds)")
            plt.ylabel("% Fast Approvals")
            plt.title("Fast Approval Rate Comparison (All Technicians)")
            plt.ylim(0, 100)
            plt.grid(alpha=0.25)
            plt.legend(title="Technician")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            rates = [(d["DURATION_SEC"] < x).mean() * 100 for x in thresholds]
            fig = plt.figure(figsize=(6, 4))
            plt.plot(thresholds, rates, marker="o", linewidth=3, color=COLOR_MAP.get(tech, "gray"))
            plt.xlabel("Threshold (seconds)")
            plt.ylabel("% Fast Approvals")
            plt.title(f"{tech} — Fast Approval Rate")
            plt.ylim(0, 100)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

# ---------------- TAB 3: Blocks / Sessions ----------------
with tab3:
    st.subheader("Blocks / Sessions")
    st.caption("A new block starts when the gap between approvals is ≥ 10 minutes (likely a new review session).")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        def build_blocks(df_tech: pd.DataFrame) -> pd.DataFrame:
            x = df_tech.sort_values("APPROVAL_DATE").copy()
            x["GAP_MIN"] = x["DURATION_SEC"] / 60
            x["NEW_BLOCK"] = x["GAP_MIN"] >= 10
            x["BLOCK_ID"] = x["NEW_BLOCK"].cumsum()
            blk = x.groupby("BLOCK_ID").agg(
                cases=("DURATION_SEC", "count"),
                avg_gap=("DURATION_SEC", "mean")
            ).reset_index()
            return blk

        if tech == "All Tech":
            for t in TECH_ORDER:
                sub = d[d["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue
                block = build_blocks(sub)

                st.markdown(f"### {t}")
                fig = plt.figure(figsize=(10, 5))
                sizes = (block["cases"] / block["cases"].max()) * 800 + 50
                plt.scatter(
                    block["cases"], block["avg_gap"],
                    s=sizes,
                    alpha=0.65,
                    color=COLOR_MAP.get(t, "gray"),
                    edgecolors="black",
                    linewidth=0.5
                )
                plt.xlabel("Number of Cases in Block")
                plt.ylabel("Average Duration Between Approvals (Seconds)")
                plt.title(f"{t} — Approvals per Block vs Average Gap (Bubble)")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            block = build_blocks(d)

            fig = plt.figure(figsize=(10, 5))
            sizes = (block["cases"] / block["cases"].max()) * 800 + 50
            plt.scatter(
                block["cases"], block["avg_gap"],
                s=sizes,
                alpha=0.65,
                color=COLOR_MAP.get(tech, "gray"),
                edgecolors="black",
                linewidth=0.5
            )
            plt.xlabel("Number of Cases in Block")
            plt.ylabel("Average Duration Between Approvals (Seconds)")
            plt.title(f"{tech} — Approvals per Block vs Average Gap (Bubble)")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)

# ---------------- TAB 4: Time patterns ----------------
with tab4:
    st.subheader("Time Patterns")
    st.caption("Weekday and hour-of-day patterns can reveal concentrated approval behavior.")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        def plot_time_panels(df_tech: pd.DataFrame, tech_name: str):
            weekday_counts = df_tech["WEEKDAY"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
            hour_counts = df_tech["HOUR_OF_DAY"].value_counts().reindex(list(range(24)), fill_value=0)

            cA, cB = st.columns(2)

            with cA:
                fig = plt.figure(figsize=(7, 4))
                plt.bar(weekday_counts.index, weekday_counts.values,
                        color=COLOR_MAP.get(tech_name, "gray"), edgecolor="black")
                plt.xlabel("Weekday")
                plt.ylabel("Approvals")
                plt.title(f"{tech_name} — Approvals by Weekday")
                plt.xticks(rotation=30)
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            with cB:
                fig = plt.figure(figsize=(7, 4))
                plt.bar(hour_counts.index, hour_counts.values,
                        color=COLOR_MAP.get(tech_name, "gray"), edgecolor="black")
                plt.xlabel("Hour of Day (0–23)")
                plt.ylabel("Approvals")
                plt.title(f"{tech_name} — Approvals by Hour")
                plt.xticks(range(0, 24, 2))
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

            st.subheader(f"{tech_name} — Weekday × Hour Heatmap")

            heat = (
                df_tech.pivot_table(index="WEEKDAY", columns="HOUR_OF_DAY", values="CASE_NUMBER",
                                    aggfunc="count", fill_value=0)
                .reindex(index=WEEKDAY_ORDER, columns=list(range(24)), fill_value=0)
            )

            fig = plt.figure(figsize=(12, 4))
            plt.imshow(heat.values, aspect="auto")
            plt.yticks(range(len(WEEKDAY_ORDER)), WEEKDAY_ORDER)
            plt.xticks(range(0, 24, 2), range(0, 24, 2))
            plt.xlabel("Hour of Day")
            plt.ylabel("Weekday")
            plt.title(f"{tech_name} — Approval Concentration (Counts)")
            plt.colorbar(label="Approvals")
            plt.tight_layout()
            st.pyplot(fig)

        if tech == "All Tech":
            for t in TECH_ORDER:
                sub = d[d["TECHNICIAN"] == t]
                if len(sub) == 0:
                    continue
                st.markdown(f"### {t}")
                plot_time_panels(sub, t)
                st.divider()
        else:
            plot_time_panels(d, tech)

# ---------------- TAB 5: High risk table ----------------
with tab5:
    st.subheader("High-Risk Fast Approvals (Worst Cases)")
    st.caption("Shows the fastest consecutive approvals in the selected filter window.")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        worst = d.sort_values("DURATION_SEC").head(50)
        st.dataframe(worst[["APPROVAL_DATE", "CASE_NUMBER", "DURATION_SEC", "TECHNICIAN"]],
                     use_container_width=True)

# ---------------- TAB 6: Policy change (Arnold only) ----------------
with tab6:
    st.subheader("Policy Change Behavior — Arnold")
    st.caption("Cutoff date: June 1, 2020 (payout reduced from $50 to $17). Only Arnold has post-policy data.")

    cutoff = pd.Timestamp("2020-06-01")
    arn = d_all[d_all["TECHNICIAN"] == "Arnold"].copy()

    if len(arn) == 0:
        st.warning("No Arnold data in the selected date range.")
    else:
        before = arn[arn["APPROVAL_DATE"] < cutoff]
        after = arn[arn["APPROVAL_DATE"] >= cutoff]

        arn["MONTH"] = arn["APPROVAL_DATE"].dt.to_period("M").dt.to_timestamp()
        arn["FAST_10S"] = arn["DURATION_SEC"] < 10

        monthly = arn.groupby("MONTH").agg(
            approvals=("DURATION_SEC", "count"),
            pct_fast=("FAST_10S", "mean")
        ).reset_index()

        fig = plt.figure(figsize=(10, 4))
        plt.plot(monthly["MONTH"], monthly["pct_fast"] * 100, marker="o", linewidth=3, color="red")
        plt.axvline(cutoff, linestyle="--", linewidth=2, color="black")
        plt.xlabel("Month")
        plt.ylabel("% Approvals < 10 sec")
        plt.title("Arnold — Fast Approval Trend Over Time")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Before vs After Summary")
        summary = pd.DataFrame({
            "Period": ["Before Policy", "After Policy"],
            "Total Approvals": [len(before), len(after)],
            "Mean Duration (sec)": [round(before["DURATION_SEC"].mean(), 2), round(after["DURATION_SEC"].mean(), 2)],
            "Median Duration (sec)": [round(before["DURATION_SEC"].median(), 2), round(after["DURATION_SEC"].median(), 2)],
            "% Fast (<10s)": [round((before["DURATION_SEC"] < 10).mean() * 100, 2),
                             round((after["DURATION_SEC"] < 10).mean() * 100, 2)]
        })
        st.dataframe(summary, use_container_width=True)
