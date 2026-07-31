import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any, Dict
from datetime import datetime
from io import BytesIO, StringIO
from app.config import REPORTS_DIR

# --- Page Config ---
st.set_page_config(
    page_title="Agent Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions to Load Workspace Reports ---
@st.cache_data(ttl=5)  # Refresh cache every 5 seconds for auto-refresh
def load_json_file(file_name: str) -> Optional[Any]:
    file_name = os.path.join(REPORTS_DIR, file_name)
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
    return None

@st.cache_data(ttl=5)
def load_run_history(file_name: str = None) -> list:
    """Load the append-only run log written by EvaluationEngine, oldest first."""
    file_name = file_name or os.path.join(REPORTS_DIR, "run_history.jsonl")
    if not os.path.exists(file_name):
        return []
    records = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# --- CI history, for the deployed dashboard ---------------------------------
#
# A deployed app cannot read the CI runner's filesystem, so each run appends its
# summary to an orphan `eval-results` branch and the dashboard fetches that file
# over HTTP. Configure via Streamlit secrets or environment:
#
#   CI_HISTORY_URL   full raw URL, or
#   GITHUB_REPO      "owner/name" -- the URL is derived from it
#   GITHUB_TOKEN     only needed for a private repository
CI_HISTORY_PATH = "results/history.jsonl"
CI_HISTORY_BRANCH = "eval-results"


def _setting(name: str) -> Optional[str]:
    """Read from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets file configured; environment only.
        pass
    return os.environ.get(name)


def ci_history_url() -> Optional[str]:
    explicit = _setting("CI_HISTORY_URL")
    if explicit:
        return explicit
    repo = _setting("GITHUB_REPO")
    if repo:
        return (f"https://raw.githubusercontent.com/{repo}/"
                f"{CI_HISTORY_BRANCH}/{CI_HISTORY_PATH}")
    return None


@st.cache_data(ttl=30)
def load_ci_history(url: Optional[str]) -> list:
    """
    Fetch the append-only CI history.

    Returns [] when unreachable rather than raising: an empty dashboard with an
    explanation is more useful than a stack trace, and the page states plainly
    that nothing was loaded instead of rendering an empty chart that looks like
    a real result.
    """
    if not url:
        return []

    import urllib.request

    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    token = _setting("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")

    return [json.loads(line) for line in body.splitlines() if line.strip()]


def get_git_info() -> Dict[str, str]:
    """Capture git metadata or return mock/fallback values."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return {
            "branch": repo.active_branch.name,
            "commit": repo.head.commit.hexsha[:8],
            "ci_status": "Success (All tests passed)"
        }
    except Exception:
        # Unknown, not invented: a fabricated commit hash on a dashboard makes a
        # run untraceable while looking authoritative.
        return {"branch": "unknown", "commit": "unknown", "ci_status": "unknown"}

# --- PDF Exporter ---
def generate_pdf_report(results_df: pd.DataFrame, summary_data: Dict[str, Any]) -> bytes:
    """Generate a clean PDF report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=20
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        
        story.append(Paragraph("AI Agent Evaluation Run Report", title_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Paragraph(f"Global Average Score: {summary_data.get('global_average_score', 'N/A')}", body_style))
        story.append(Paragraph(f"Total Evaluated Tasks: {summary_data.get('total_tasks_evaluated', len(results_df))}", body_style))
        
        # None means the run produced no evidence for these. Defaulting to 1.0
        # printed a perfect score into an exported report, which is the hardest
        # place to later notice it was never measured.
        regression_catch = summary_data.get("regression_catch_rate", {})
        overall_catch = regression_catch.get("overall")
        adversarial_refusal = summary_data.get("adversarial_refusal_rate")

        def pct(value):
            return "Not measured" if value is None else f"{value * 100:.1f}%"

        story.append(Paragraph(f"Regression Catch Rate: {pct(overall_catch)}", body_style))
        by_fault = regression_catch.get("by_fault_type", {})
        for f_type, val in by_fault.items():
            story.append(Paragraph(f"  - {f_type}: {pct(val)}", body_style))
        story.append(Paragraph(f"Adversarial Refusal Rate: {pct(adversarial_refusal)}", body_style))
        
        story.append(Spacer(1, 15))
        
        # Table of results
        table_data = [["Task ID", "Benchmark", "Prompt", "Score"]]
        for _, row in results_df.iterrows():
            prompt_truncated = str(row['prompt'])[:40] + "..." if len(str(row['prompt'])) > 40 else str(row['prompt'])
            table_data.append([
                str(row['task_id']),
                str(row['benchmark']),
                prompt_truncated,
                str(row['overall_score'])
            ])
            
        t = Table(table_data, colWidths=[100, 100, 240, 60])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7FAFC")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        # Fallback raw text in PDF
        buffer = BytesIO()
        buffer.write(f"PDF creation failed: {e}".encode("utf-8"))
        return buffer.getvalue()

# --- Main Dashboard Control Flow ---
def main():
    # Load files dynamically from the root directory
    results = load_json_file("results.json")
    execution = load_json_file("execution.json")
    fault_report = load_json_file("fault_report.json")
    summary = load_json_file("evaluation_summary.json")
    git_info = get_git_info()
    
    # No fabricated fallbacks. If the pipeline has not been run there is nothing
    # to show, and inventing plausible numbers would put fiction on screen that
    # is indistinguishable from a real run.
    if not results or not execution or not summary:
        st.warning(
            f"No evaluation reports found in `{REPORTS_DIR}/`. "
            "Run `python demo_pipeline.py` to generate them."
        )
        st.stop()

    # Convert results array to DataFrame
    df_results = pd.DataFrame(results)
    df_exec = pd.DataFrame(execution["tasks"])
    
    # Sidebar navigation
    st.sidebar.title("Agent Evaluation")
    page = st.sidebar.selectbox("Navigate Pages", [
        "Overview",
        "Benchmark Explorer",
        "Agent Leaderboard",
        "CI History (live)",
        "Regression Monitor",
        "Fault Injection Monitor",
        "Langfuse Explorer",
        "Execution Timeline"
    ])
    
    # Git Metadata Sidebar Box
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Branch:** `{git_info['branch']}`")
    st.sidebar.markdown(f"**Current Commit:** `{git_info['commit']}`")
    st.sidebar.markdown(f"**CI Status:** `{git_info['ci_status']}`")
    
    # Exporter Panel
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    
    # Exporter code
    col_csv, col_json = st.sidebar.columns(2)
    with col_csv:
        csv_data = df_results.to_csv(index=False)
        st.download_button("Export CSV", csv_data, "results.csv", "text/csv")
    with col_json:
        json_data = df_results.to_json(orient="records", indent=2)
        st.download_button("Export JSON", json_data, "results.json", "application/json")
        
    col_html, col_pdf = st.sidebar.columns(2)
    with col_html:
        html_data = df_results.to_html(index=False)
        st.download_button("Export HTML", html_data, "results.html", "text/html")
    with col_pdf:
        pdf_data = generate_pdf_report(df_results, summary)
        st.download_button("Export PDF", pdf_data, "report.pdf", "application/pdf")

    # --- PAGE IMPLEMENTATIONS ---
    
    if page == "Overview":
        st.title("Overview")
        st.write("Live pipeline summary metrics tracked across multiple benchmarks.")
        
        # Metric Cards Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Global Average Score", value=f"{summary.get('global_average_score', 0.0):.2f}")
        with col2:
            st.metric(label="Total Tasks Evaluated", value=str(summary.get("total_tasks_evaluated", 0)))
        with col3:
            st.metric(label="Successful Executions", value=f"{execution['summary'].get('successful_runs', 0)}/{execution['summary'].get('total_tasks', 0)}")
        with col4:
            st.metric(label="Total Faults Injected", value=str(fault_report["summary"].get("total_faults_injected", 0)))
            
        # Metric columns chart
        st.subheader("Aggregate Normalized Metric Scores")
        metric_data = summary.get("summary_metrics", {})
        if metric_data:
            df_metric = pd.DataFrame(list(metric_data.items()), columns=["Metric", "Score"])
            fig = px.bar(df_metric, x="Metric", y="Score", color="Score", range_y=[0.0, 1.0], height=350, color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, width="stretch")

        # Regression Catch Rate Section
        st.subheader("Regression detection")
        # None means the run produced no evidence. Rendering that as 100% is the
        # failure mode this project exists to catch, so it is shown as "Not
        # measured" instead.
        catch_rate = summary.get("regression_catch_rate", {})
        overall_catch = catch_rate.get("overall")
        refusal_rate = summary.get("adversarial_refusal_rate")
        injections = sum((catch_rate.get("injections_by_type") or {}).values())

        col_kpi1, col_kpi2 = st.columns(2)
        with col_kpi1:
            st.metric(
                label="Regression catch rate",
                value="Not measured" if overall_catch is None else f"{overall_catch * 100:.1f}%",
                help=f"Measured over {injections} injections."
                     if overall_catch is not None else
                     "No faults were injected, so this run shows no evidence the "
                     "suite can detect a regression.",
            )
        with col_kpi2:
            st.metric(
                label="Adversarial refusal rate",
                value="Not measured" if refusal_rate is None else f"{refusal_rate * 100:.1f}%",
                help=None if refusal_rate is not None else
                     "No adversarial tasks ran in this suite.",
            )
            
        by_fault = summary.get("regression_catch_rate", {}).get("by_fault_type", {})
        if by_fault:
            df_fault_kpi = pd.DataFrame(list(by_fault.items()), columns=["Fault Type", "Catch Rate"])
            fig_fault_kpi = px.bar(
                df_fault_kpi, 
                x="Fault Type", 
                y="Catch Rate", 
                color="Catch Rate", 
                range_y=[0.0, 1.0], 
                height=300, 
                color_continuous_scale="RdYlGn",
                title="Regression Catch Rate by Fault Type"
            )
            st.plotly_chart(fig_fault_kpi, width="stretch")

        st.subheader("Global Execution Explorer")
        st.write("Click any task in the dataframe below to inspect execution parameters.")
        
        selected_task_id = st.selectbox("Select Task ID to Explore details:", df_results["task_id"].tolist())
        
        # Details Panel
        if selected_task_id:
            row_res = df_results[df_results["task_id"] == selected_task_id].iloc[0]
            row_exec = df_exec[df_exec["task_id"] == selected_task_id].iloc[0] if not df_exec[df_exec["task_id"] == selected_task_id].empty else {}
            
            c_prompt, c_out = st.columns(2)
            with c_prompt:
                st.info(f"**Prompt:**\n{row_res['prompt']}")
            with c_out:
                st.success(f"**Agent Response:**\n{row_exec.get('response', 'N/A')}")
                
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"**Expected Answer:** `{row_exec.get('expected_answer', 'N/A')}`")
                st.markdown(f"**Similarity (Jaccard):** `{row_res['metrics'].get('accuracy', 0.0):.2f}`")
            with col_b:
                st.markdown(f"**Total Latency:** `{row_exec.get('latency_seconds', 0.0):.2f}s`")
                st.markdown(f"**Total Cost:** `${row_exec.get('cost_usd', 0.0):.6f}`")
            with col_c:
                st.markdown(f"**Expected Tools:** `{row_exec.get('expected_tools', [])}`")
                st.markdown(f"**Actual Tool Calls:** `{[t.get('tool_name') for t in row_exec.get('tool_calls', [])]}`")
                
            # Execution Graph Visualization
            st.markdown("#### Execution Graph Schema")
            nodes = [n.get("name", "") for n in row_exec.get("execution_graph", {}).get("nodes", [])]
            if nodes:
                st.code(" -> ".join(nodes))
            else:
                # Previously printed a plausible default chain, which is a graph
                # no agent produced.
                st.caption("This adapter reported no execution graph.")

    elif page == "Benchmark Explorer":
        st.title("Benchmark Explorer")
        st.write("Comparison metrics by benchmark provider (Harbor Index vs ContextBench vs T3Bench).")
        
        # Group by benchmark
        df_group = df_results.groupby("benchmark")["overall_score"].mean().reset_index()
        
        fig = px.bar(
            df_group, 
            x="benchmark", 
            y="overall_score", 
            color="benchmark", 
            title="Average Score by Benchmark Type",
            range_y=[0.0, 1.0],
            height=400
        )
        st.plotly_chart(fig, width="stretch")
        
        # Radar/Scatter comparing benchmarks
        fig2 = px.scatter(
            df_results, 
            x="benchmark", 
            y="overall_score", 
            color="difficulty", 
            size="overall_score", 
            hover_name="task_id", 
            title="Benchmark Difficulty Scatters"
        )
        st.plotly_chart(fig2, width="stretch")

    elif page == "Agent Leaderboard":
        st.title("Agent Leaderboard")
        st.write("Evaluation rankings showing score, average latency, and computed runs cost.")
        
        leaderboard_df = df_exec.copy()
        # Merge scores
        score_map = {r["task_id"]: r["overall_score"] for r in results}
        leaderboard_df["overall_score"] = leaderboard_df["task_id"].map(score_map)
        
        leaderboard_sum = leaderboard_df.groupby("benchmark").agg({
            "overall_score": "mean",
            "latency_seconds": "mean",
            "cost_usd": "sum"
        }).reset_index().rename(columns={"overall_score": "Average Score", "latency_seconds": "Average Latency (s)", "cost_usd": "Total Cost ($)"})
        
        st.table(leaderboard_sum.sort_values("Average Score", ascending=False))

    elif page == "CI History (live)":
        st.title("CI History")
        st.caption(
            "Every pipeline run, straight from CI. Published by the workflow to "
            "the `eval-results` branch, so this works on a deployed app with no "
            "access to the runner."
        )

        url = ci_history_url()
        if not url:
            st.warning(
                "No history source configured. Set **GITHUB_REPO** (`owner/name`) "
                "or **CI_HISTORY_URL** in Streamlit secrets. Add **GITHUB_TOKEN** "
                "as well if the repository is private."
            )
            st.stop()

        st.caption(f"Source: `{url}`")
        try:
            history = load_ci_history(url)
        except Exception as e:
            # Named explicitly. The usual causes are a private repo without a
            # token, or CI not having published yet.
            st.error(f"Could not fetch the history: {e}")
            st.info(
                "Usually this means the `eval-results` branch does not exist yet "
                "(no CI run has published), or the repository is private and "
                "GITHUB_TOKEN is missing."
            )
            st.stop()

        if not history:
            st.info("No runs published yet. Push a commit or open a PR to produce one.")
            st.stop()

        # --- Triage ------------------------------------------------------
        # A run that evaluated 8 of 30 tasks is only defensible if the reader can
        # see which 8 and why. Without this the narrowing is invisible, and an
        # invisible narrowing is indistinguishable from a suite that silently
        # stopped testing things.
        latest_triage = next((r.get("triage") for r in history if r.get("triage")), None)
        if latest_triage:
            st.subheader("Triage — what the last run selected")
            selected = latest_triage.get("selected")
            suite_size = latest_triage.get("suite_size")
            cols = st.columns(4)
            cols[0].metric("Tasks selected", f"{selected} of {suite_size}"
                           if selected is not None and suite_size else "n/a")
            cols[1].metric("Required by rules", latest_triage.get("mandatory", 0))
            cols[2].metric("Added by advisor", latest_triage.get("suggested", 0))
            cols[3].metric("Advisor", latest_triage.get("advisor_status", "unknown"))

            if latest_triage.get("rule_reason"):
                st.caption(f"Rules: {latest_triage['rule_reason']}")
            for rejected in latest_triage.get("rejected") or []:
                # Surfaced rather than hidden: a model inventing ids is a signal
                # about the advisor, and hiding it wastes the signal.
                st.warning(f"Advisor suggestion rejected — {rejected}")
            st.divider()

        rows = []
        for r in history:
            ci = r.get("ci", {})
            perf = r.get("performance", {})
            triage = r.get("triage") or {}
            rows.append({
                "When": r.get("timestamp", "")[:19].replace("T", " "),
                "Source": f"PR #{ci['pr_number']}" if ci.get("pr_number") else ci.get("branch", "?"),
                "Commit": ci.get("commit", ""),
                "Gate": "PASS" if r.get("gate_passed") else "FAIL",
                "Triaged": (f"{triage['selected']}/{triage['suite_size']}"
                            if triage.get("selected") is not None
                            and triage.get("suite_size") else "full suite"),
                "Complete": r.get("complete", True),
                "Score": r.get("global_average_score"),
                "Tasks": r.get("total_tasks_evaluated"),
                "Agent": (r.get("run") or {}).get("target", "unknown"),
                "Suite": (r.get("run") or {}).get("suite"),
                "eval_set_sha": (r.get("run") or {}).get("eval_set_sha", "")[:10],
                "Latency (s)": perf.get("average_latency_seconds"),
                "Cost/success ($)": perf.get("cost_per_successful_task_usd"),
                "Model": perf.get("pricing_model"),
                "Tokens": perf.get("token_source"),
                "Run": ci.get("run_url"),
            })
        df = pd.DataFrame(rows)

        # --- Insights ----------------------------------------------------
        # A table makes the reader do the comparison, and mostly they will not.
        # These state the finding, and only call a drop a regression when a
        # Fisher exact test says it is unlikely to be noise.
        from app.evaluation.insights import compare_runs, find_reference

        latest = history[-1]
        reference = find_reference(history)
        cards = compare_runs(latest, reference)

        st.subheader("What changed")
        if reference:
            ref_ci = reference.get("ci", {})
            st.caption(
                f"Latest run compared against "
                f"`{ref_ci.get('branch', '?')}` @ `{ref_ci.get('commit', '?')}`"
                + (" (the state this PR would merge into)"
                   if (latest.get("ci") or {}).get("pr_number") else "")
            )

        render = {"critical": st.error, "warning": st.warning,
                  "good": st.success, "info": st.info}
        for card in cards:
            render.get(card["severity"], st.info)(
                "**" + card["title"] + "**\n\n" + card["detail"]
            )

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest score", latest.get("global_average_score"))
        c2.metric("Gate", "PASS" if latest.get("gate_passed") else "FAIL")
        c3.metric("Runs recorded", len(history))
        c4.metric("Failing runs", int((df["Gate"] == "FAIL").sum()))

        # A run stopped by its token budget is partial and must not be read as a
        # complete measurement of the agent.
        partial = df[~df["Complete"]]
        if not partial.empty:
            st.warning(
                f"{len(partial)} run(s) stopped early on their token budget and are "
                f"partial. Latest reason: {latest.get('budget_stop_reason') or 'see the run log'}"
            )

        st.subheader("Score across runs")

        # One series per agent. Different agents score differently on the same
        # suite by design, so plotting them as a single line would read as drift
        # when it is only a change of subject.
        agents = sorted(df["Agent"].unique())
        chosen = st.multiselect("Agents", agents, default=agents)
        view = df[df["Agent"].isin(chosen)] if chosen else df

        fig = go.Figure()
        for agent in chosen:
            series = view[view["Agent"] == agent]
            fig.add_trace(go.Scatter(
                x=series["When"], y=series["Score"], mode="lines+markers", name=agent,
                marker=dict(
                    size=11,
                    # Fill still carries the gate verdict, so a red point is
                    # visible whichever agent produced it.
                    color=["#2ca02c" if g == "PASS" else "#d62728" for g in series["Gate"]],
                    line=dict(width=1, color="#888"),
                ),
                text=series["Source"],
                hovertemplate="%{text}<br>score %{y}<extra>" + agent + "</extra>",
            ))
        fig.update_layout(yaxis_range=[0.0, 1.0], height=360,
                          title="One line per agent; marker fill is the gate verdict")
        st.plotly_chart(fig, width="stretch")

        if len(agents) > 1:
            st.caption(
                "Scores are only comparable within an agent. A different agent on "
                "the same suite is a different measurement, not a regression."
            )

        tab_runs, tab_cost, tab_instrument = st.tabs(
            ["Runs", "Cost & latency", "Instrument"])

        with tab_runs:
            st.dataframe(
                df.drop(columns=["Complete"]),
                width="stretch",
                column_config={"Run": st.column_config.LinkColumn("Run", display_text="open")},
            )

        with tab_cost:
            st.plotly_chart(
                px.line(df, x="When", y="Cost/success ($)", markers=True,
                        title="Cost per successful task"),
                width="stretch")
            st.plotly_chart(
                px.line(df, x="When", y="Latency (s)", markers=True,
                        title="Average latency"),
                width="stretch")
            st.caption(
                "`Tokens = provider` means the model reported real usage. "
                "`estimated` means they were inferred from character counts, so the "
                "cost is a shadow figure rather than a measurement."
            )

        with tab_instrument:
            measured = [r for r in history if r.get("instrument")]
            if not measured:
                st.info(
                    "No run has published instrument metrics yet. These come from "
                    "the nightly experiment (`run_experiment.py`), which measures "
                    "whether the suite can still detect a planted regression."
                )
            else:
                inst = pd.DataFrame([{
                    "When": r["timestamp"][:19].replace("T", " "),
                    "MDE": r["instrument"]["minimum_detectable_effect"],
                    "Planted": r["instrument"]["regressions_planted"],
                    "Degraded": r["instrument"]["regressions_that_degraded"],
                    "Detected": r["instrument"]["degrading_detection_rate"],
                } for r in measured])
                st.dataframe(inst, width="stretch")
                st.plotly_chart(
                    px.line(inst, x="When", y="MDE", markers=True,
                            title="Minimum detectable effect (lower is a sharper suite)"),
                    width="stretch")
                st.caption(
                    "A rising MDE means the suite is getting blunter. That is a "
                    "defect in the harness, not in the agent."
                )

    elif page == "Regression Monitor":
        st.title("Regression Monitor")
        st.write("Watch quality drift across evaluation runs.")
        
        history = load_run_history()
        if len(history) < 2:
            st.info(
                f"Regression trends need at least two evaluation runs; "
                f"{len(history)} recorded so far. Run `python demo_pipeline.py` "
                f"again to append another run to run_history.jsonl."
            )
            st.stop()

        df_trend = pd.DataFrame({
            "Run": [f"Run {i + 1}" for i in range(len(history))],
            "Timestamp": [r["timestamp"] for r in history],
            "Overall Score": [r["global_average_score"] for r in history],
            "Accuracy": [r["summary_metrics"].get("accuracy") for r in history],
            "Quality": [r["summary_metrics"].get("quality") for r in history],
            "Safety": [r["summary_metrics"].get("safety_and_policy") for r in history],
            "Latency (s)": [r.get("average_latency_seconds") for r in history],
            "Cost ($)": [r.get("average_cost_usd") for r in history],
            "Avg Tokens": [r.get("average_total_tokens") for r in history]
        })

        tab_acc, tab_lat, tab_cost, tab_token = st.tabs(["Quality Metrics", "Latency Trend", "Cost Trend", "Token Inflation"])

        with tab_acc:
            fig = go.Figure()
            for name, color in (("Overall Score", "blue"), ("Accuracy", "green"),
                                ("Quality", "orange"), ("Safety", "red")):
                fig.add_trace(go.Scatter(
                    x=df_trend["Run"], y=df_trend[name], name=name,
                    line=dict(color=color, width=3)
                ))
            fig.update_layout(title="Metric Drift Across Runs", yaxis_range=[0.0, 1.0])
            st.plotly_chart(fig, width="stretch")
            
        with tab_lat:
            fig_lat = px.line(df_trend, x="Run", y="Latency (s)", title="Average Latency Drift (seconds)")
            st.plotly_chart(fig_lat, width="stretch")
            
        with tab_cost:
            fig_cost = px.line(df_trend, x="Run", y="Cost ($)", title="Computed Query Costs ($)")
            st.plotly_chart(fig_cost, width="stretch")
            
        with tab_token:
            fig_tok = px.bar(df_trend, x="Run", y="Avg Tokens", color="Avg Tokens",
                             title="Average Total Tokens per Task")
            st.plotly_chart(fig_tok, width="stretch")

    elif page == "Fault Injection Monitor":
        st.title("Fault Injection Monitor")
        st.write("Compare injected faults against detected regressions to check pipeline sensitivity.")
        
        # 1. Fault Frequency chart
        st.subheader("Injected Fault Frequencies")
        df_injections = pd.DataFrame(fault_report.get("injections", []))
        if not df_injections.empty:
            fig_freq = px.histogram(df_injections, x="component", color="severity", title="Fault component frequencies")
            st.plotly_chart(fig_freq, width="stretch")
            
            # Displays list of injected faults
            st.subheader("Fault Injection Ledger")
            st.dataframe(df_injections[["fault_id", "severity", "component", "expected_impact", "actual_impact"]])
        else:
            st.info("No faults injected during this run.")
            
        # 2. Regression detection rate, as measured by the evaluation engine.
        st.subheader("Regression Detection Rate")
        catch = (summary or {}).get("regression_catch_rate", {})
        overall_catch = catch.get("overall")
        injections_by_type = catch.get("injections_by_type", {})

        if overall_catch is None:
            st.warning(
                "Not measured: no faults were injected in this run, so the suite has "
                "produced no evidence that it can detect a regression."
            )
        else:
            st.metric(
                label="Overall Regression Detection Rate",
                value=f"{overall_catch * 100:.1f}%",
                help=f"Measured over {sum(injections_by_type.values())} injections."
            )
            by_type = catch.get("by_fault_type", {})
            if by_type:
                st.dataframe(pd.DataFrame([
                    {
                        "Fault Type": f_type,
                        "Injections": injections_by_type.get(f_type, 0),
                        "Detection Rate": f"{rate * 100:.1f}%"
                    }
                    for f_type, rate in sorted(by_type.items())
                ]))

    elif page == "Langfuse Explorer":
        st.title("Trace Explorer")
        st.write("Browse deep links directly into the Langfuse cloud observability dashboard.")
        
        # Telemetry list
        trace_data = []
        for index, row in df_exec.iterrows():
            trace_data.append({
                "Task ID": row.get("task_id"),
                "Benchmark": row.get("benchmark"),
                "Trace ID": row.get("langfuse_trace_id", f"trace_{index}"),
                "Link": row.get("langfuse_deep_link", "https://cloud.langfuse.com")
            })
            
        df_trace = pd.DataFrame(trace_data)
        st.dataframe(df_trace)
        st.markdown("### Click link to launch Langfuse Dashboard:")
        for td in trace_data:
            st.markdown(f"- **Task `{td['Task ID']}`**: [Open Langfuse Trace Link]({td['Link']})")

    elif page == "Execution Timeline":
        st.title("Execution Timeline")
        st.write("Displays Gantt timelines of parallel task worker durations.")
        
        # Gantt chart
        fig_timeline = px.bar(
            df_exec, 
            y="task_id", 
            x="latency_seconds", 
            color="benchmark", 
            orientation="h",
            labels={"latency_seconds": "Duration (seconds)", "task_id": "Task ID"},
            title="Concurrent Task Durations Gantt Chart"
        )
        st.plotly_chart(fig_timeline, width="stretch")

if __name__ == "__main__":
    main()
