with col_gi:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚠️ GI Adverse Event Rate (%)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_h']) // 800)
        fig_gi.add_trace(go.Scatter(
            x=r['t_h'][::s], y=r['GI'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2.5,
                      dash=COHORT_DASH[coh]),
            hovertemplate=(
                f"<b>{coh}</b><br>"
                "Time: %{x:.0f} h<br>"
                "GI AE: %{y:.2f}%<extra></extra>"
            )
        ))
    fig_gi.update_layout(
        **CHART_BG, height=340,
        xaxis_title="Time (h)",
        yaxis_title="GI AE rate (%)",
        showlegend=False
    )
    fig_gi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_gi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
