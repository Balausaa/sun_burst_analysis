from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

from backend.processing.surya_processor import SuryaAnalysisResult, analyze_with_surya
from backend.risk_engine import assess_surya_risk
from backend.synthetic_generator import SyntheticConfig, generate_synthetic_signal
from backend.models.surya_model import SuryaWrapper
from backend.processing.signal_processor import analyze_signal_matrix
from backend.hybrid_analyzer import CombinedAnalysisResult, analyze_csv_combined


st.set_page_config(
    page_title="AI spacewhether - Space Weather Early Warning",
    layout="wide",
)


def _plot_spectrogram(result: SuryaAnalysisResult) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=result.spectrogram.T,
            x=result.times,
            y=result.freqs,
            colorscale="Viridis",
            colorbar=dict(title="Intensity (a.u.)"),
        )
    )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (MHz)",
        title="Dynamic Spectrum",
    )
    return fig


def _gauge(title: str, value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100.0,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _risk_color(level: str) -> str:
    if level == "EXTREME":
        return "#8b0000"
    if level == "HIGH":
        return "#ff4b4b"
    if level == "MEDIUM":
        return "#faca2b"
    return "#3ac569"


def _flare_color(p: float) -> str:
    if p < 0.3:
        return "#3ac569"
    if p < 0.7:
        return "#faca2b"
    return "#ff4b4b"


def _render_flare_progress(prob: float) -> None:
    pct = float(np.clip(prob, 0.0, 1.0)) * 100.0
    color = _flare_color(prob)
    st.markdown(
        f"""
        <div style="width:100%;background-color:#e0e0e0;border-radius:8px;overflow:hidden;">
          <div style="width:{pct:.1f}%;background-color:{color};height:20px;"></div>
        </div>
        <div style="margin-top:4px;"><b>Flare Probability:</b> {prob*100:.2f}%</div>
        """,
        unsafe_allow_html=True,
    )


def _analyze_bytes(file_bytes: bytes):
    """
    Run the Surya-based pipeline and the Type II (CNN+LSTM) pipeline
    on the same spectrogram so we can report both radio burst metrics
    and the hybrid final satellite risk used by the API.
    """
    # Surya analysis for spectrogram and flare/solar wind metrics.
    analysis: SuryaAnalysisResult = analyze_with_surya(file_bytes)
    # Type II module on the same spectrogram.
    t2_result = analyze_signal_matrix(analysis.spectrogram)
    # Hybrid combined risk (matches the /predict API behaviour).
    combined: CombinedAnalysisResult = analyze_csv_combined(file_bytes)
    return analysis, t2_result, combined


def main() -> None:
    st.title("AI spacewhether")
    st.subheader("Surya-powered Space Weather Early Warning (MVP)")

    st.markdown(
        "AI spacewhether provides automated space weather risk assessment by "
        "processing solar radio burst data, computing statistical signal features, "
        "and applying a trained ML classification model to estimate flare probability."
    )

    with st.sidebar:
        st.header("Input")
        satellite_id = st.text_input("Satellite ID", value="HELIO-SAT-1")
        uploaded = st.file_uploader(
            "Upload dynamic spectrum CSV (`time,frequency,intensity`)", type=["csv"]
        )

    col_left, col_right = st.columns([2, 1])

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if uploaded is not None:
        content = uploaded.read()
        analysis, t2_result, combined = _analyze_bytes(content)

        with col_left:
            st.plotly_chart(_plot_spectrogram(analysis), use_container_width=True)
            st.markdown("### Signal Metrics")
            st.write(
                {
                    "Entropy": round(analysis.entropy, 3),
                    "Drift rate (MHz/s)": round(analysis.drift_rate, 3),
                    "Max intensity": round(analysis.max_intensity, 3),
                    "Mean intensity": round(analysis.mean_intensity, 3),
                }
            )

        with col_right:
            st.markdown("### Radio Burst Analysis")
            if t2_result is not None:
                st.plotly_chart(
                    _gauge(
                        "Type II Probability",
                        t2_result.type_ii_probability,
                        color="#ff7f50",
                    ),
                    use_container_width=True,
                )
                st.write(f"Drift Rate: {t2_result.drift_rate:.2f} MHz/s")

            st.markdown("### Space Weather Forecast")
            st.plotly_chart(
                _gauge(
                    "Flare Probability",
                    analysis.flare_probability,
                    color="#4b9cff",
                ),
                use_container_width=True,
            )
            st.metric("Solar Wind Speed (km/s)", f"{analysis.solar_wind_speed:.0f}")
            st.write(f"Geomagnetic Risk: {combined.geomagnetic_risk}")
            st.write(f"Final Satellite Risk: {combined.final_risk_level}")

            banner_color = _risk_color(combined.final_risk_level)
            st.markdown(
                f"<div style='padding:1rem;border-radius:0.5rem;background:{banner_color};color:white;'>"
                f"<b>Final Satellite Risk: {combined.final_risk_level}</b><br/>{combined.recommendation}"
                f"</div>",
                unsafe_allow_html=True,
            )

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "satellite_id": satellite_id,
            "type_ii_probability": float(t2_result.type_ii_probability) if t2_result is not None else 0.0,
            "flare_probability": analysis.flare_probability,
            "solar_wind_speed": analysis.solar_wind_speed,
            "entropy": analysis.entropy,
            "drift_rate": analysis.drift_rate,
            "max_intensity": analysis.max_intensity,
            "mean_intensity": analysis.mean_intensity,
            "geomagnetic_risk": combined.geomagnetic_risk,
            "final_risk_level": combined.final_risk_level,
            "risk_level": combined.final_risk_level,
            "advice": combined.recommendation,
        }
        st.session_state["history"].append(event)

        # Always show details for the most recent event after a new CSV is processed.
        st.markdown("### Latest Event Details")
        st.write(
            {
                "Timestamp": event["timestamp"],
                "Satellite ID": event["satellite_id"],
                "Flare Probability": event["flare_probability"],
                "Solar Wind Speed (km/s)": event["solar_wind_speed"],
                "Entropy": event["entropy"],
                "Drift rate": event["drift_rate"],
                "Max intensity": event["max_intensity"],
                "Mean intensity": event["mean_intensity"],
                "Risk level": event["risk_level"],
                "Advice": event["advice"],
            }
        )

    st.markdown("---")
    st.markdown("### Event History (Session)")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        hist_df = hist_df.sort_values("timestamp", ascending=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=hist_df["timestamp"],
                y=hist_df["flare_probability"],
                mode="lines+markers",
                name="Flare Probability",
            )
        )
        fig_hist.update_layout(
            xaxis_title="Time",
            yaxis_title="Flare Probability",
            title="Historical Flare Probability",
        )

        clicked = plotly_events(fig_hist, click_event=True, hover_event=False)

        st.dataframe(
            hist_df.sort_values("timestamp", ascending=False), use_container_width=True
        )

        if clicked:
            # Use the x-coordinate (timestamp) from the clicked point to
            # look up the corresponding event, which is more robust than
            # relying on pointIndex alone.
            clicked_ts = str(clicked[0].get("x"))
            row = hist_df[hist_df["timestamp"] == clicked_ts].iloc[-1]
            st.markdown("### Selected Event Details")
            st.write(
                {
                    "Timestamp": row["timestamp"],
                    "Satellite ID": row.get("satellite_id", ""),
                    "Flare Probability": row["flare_probability"],
                    "Solar Wind Speed (km/s)": row["solar_wind_speed"],
                    "Entropy": row["entropy"],
                    "Drift rate": row["drift_rate"],
                    "Max intensity": row["max_intensity"],
                    "Mean intensity": row["mean_intensity"],
                    "Risk level": row["risk_level"],
                    "Advice": row["advice"],
                }
            )
    else:
        st.info("Run an analysis to populate the event history.")


if __name__ == "__main__":
    main()

