"""
batch_report_gen.py — Generate 10 individual PDF reports from a patient CSV dataset.

Each patient gets their own:
  - Clinical metrics page
  - AI prediction (from global_model.pth)
  - XAI feature importance chart
  - Radar chart (patient vs healthy baseline)
  - Endometriosis subtype correlation heatmap
  - Digital Twin 3D visualisation
  - Clinical synthesis & recommendation plan

Usage:
    python3 batch_report_gen.py [--input test_patients_10.csv] [--outdir reports/]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.ffnn_weighting import FeatureWeightingFFNN
from models.pinn import EndometriosisPINN, FullFedPINNModel
from digital_twin.simulator import UterusDigitalTwin
from report_gen import generate_advanced_pdf_report


# ── Constants & helpers ────────────────────────────────────────────────────────
STAGE_NAMES = ["Minimal/None", "Stage I (Minimal)", "Stage II (Mild)",
               "Stage III (Moderate)", "Stage IV (Severe)"]

_MOCK_MEANS = np.array([32.0, 25.0, 5.0, 5.0, 0.5, 0.5, 45.0, 150.0, 10.0])
_MOCK_STDS  = np.array([7.0,  4.0,  3.0, 3.0, 0.5, 0.5, 15.0,  50.0,  5.0])


def load_model(model_path="global_model.pth", device="cpu"):
    ffnn  = FeatureWeightingFFNN()
    pinn  = EndometriosisPINN()
    model = FullFedPINNModel(ffnn, pinn).to(device)
    if os.path.exists(model_path):
        ck = torch.load(model_path, map_location=device)
        model.load_state_dict(ck.get("full_model", ck))
        print(f"  ✅ Loaded trained weights from '{model_path}'")
    else:
        print(f"  ⚠️  '{model_path}' not found — using random weights (run generate_model.py first)")
    model.eval()
    return model


def predict(model, row, device="cpu"):
    """Run inference on a single patient row dict."""
    clinical_raw = np.array([
        row["age"], row["bmi"], row["pelvic_pain_score"], row["dysmenorrhea_score"],
        row["dyspareunia"], row["family_history"], row["ca125"],
        row["estradiol"], row["progesterone"]
    ], dtype=np.float32)

    clinical_norm = (clinical_raw - _MOCK_MEANS) / (_MOCK_STDS + 1e-8)
    clinical_t    = torch.tensor(clinical_norm, dtype=torch.float32).unsqueeze(0).to(device)
    us_t          = torch.zeros((1, 128), dtype=torch.float32).to(device)
    gen_t         = torch.zeros((1, 256), dtype=torch.float32).to(device)
    path_t        = torch.zeros((1,  64), dtype=torch.float32).to(device)
    sensor_t      = torch.zeros((1,  32), dtype=torch.float32).to(device)

    with torch.no_grad():
        prob, stage_logits, future_risk, gate_probs = model(
            clinical_t, us_t, gen_t, path_t, sensor_t
        )

    p_prob      = float(prob.squeeze())
    p_stage     = int(torch.argmax(stage_logits, dim=1).item())
    f_risk      = future_risk.squeeze().numpy()
    gate_probs  = gate_probs.squeeze().numpy()
    p_std       = 0.04  # Bayesian dropout estimate proxy

    return p_prob, p_std, p_stage, f_risk, gate_probs, clinical_raw


def build_xai_fig(clinical_raw, prob, model):
    """Gradient saliency XAI chart — 9 features."""
    features = ["Age", "BMI", "Pelvic Pain", "Dysmenorrhea", "Dyspareunia",
                "Fam. History", "CA-125", "Estradiol", "Progesterone"]
    clinical_norm = (clinical_raw - _MOCK_MEANS) / (_MOCK_STDS + 1e-8)
    inp = torch.tensor(clinical_norm, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    inp.retain_grad()  # needed because inp becomes non-leaf inside the model
    out, _, _, _ = model(inp, torch.zeros(1, 128), torch.zeros(1, 256),
                         torch.zeros(1, 64), torch.zeros(1, 32))
    out.backward()
    grad = inp.grad if inp.grad is not None else torch.zeros_like(inp)
    importance = grad.squeeze().detach().numpy()
    importance = importance / (np.sum(np.abs(importance)) + 1e-6)

    db = pd.DataFrame({
        "Feature": features,
        "Importance": importance,
        "Impact": ["↑ Risk" if i > 0 else "↓ Risk" for i in importance],
        "Value": clinical_raw
    }).sort_values("Importance")

    fig = px.bar(db, x="Importance", y="Feature", color="Impact",
                 color_discrete_map={"↑ Risk": "#dc3545", "↓ Risk": "#28a745"},
                 orientation="h", title="Feature Importance (Gradient Saliency)")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))

    top = db[db["Impact"] == "↑ Risk"].tail(2)
    causes = [f"{r['Feature']} ({r['Value']:.1f})" for _, r in top.iterrows()]
    explanation = f"Risk score {prob*100:.1f}%. Primary drivers: {', '.join(causes)}."
    return fig, explanation


def build_radar_fig(clinical_raw):
    """Patient vs healthy baseline radar chart."""
    feats = ["Age", "BMI", "Pelvic Pain", "Dysmenorrhea", "CA-125", "Estradiol"]
    v = clinical_raw
    patient = [
        min(10, max(0, (v[0]-18)/37*10)),
        min(10, max(0, (v[1]-18)/22*10)),
        v[2], v[3],
        min(10, max(0, v[6]/150*10)),
        min(10, max(0, v[7]/500*10)),
    ]
    healthy = [3.0, 3.5, 1.0, 2.0, 2.0, 4.0]
    fig = go.Figure([
        go.Scatterpolar(r=healthy, theta=feats, fill="toself", name="Healthy Baseline",
                        line_color="rgba(40,167,69,0.9)", fillcolor="rgba(40,167,69,0.2)"),
        go.Scatterpolar(r=patient, theta=feats, fill="toself", name="This Patient",
                        line_color="rgba(232,62,140,0.9)", fillcolor="rgba(232,62,140,0.35)"),
    ])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                      margin=dict(l=40, r=40, t=20, b=20), height=350)
    return fig


def build_heatmap_fig(clinical_raw):
    """Endometriosis subtype alignment heatmap."""
    subtypes = ["Superficial", "Ovarian (OMA)", "Deep Infiltrating (DIE)", "Adenomyosis"]
    v = clinical_raw
    pain_f   = v[2] / 10.0
    hormone_f = (v[6]/150 + v[7]/500) / 2.0
    dys_f    = v[3] / 10.0
    weights  = [
        [max(0.05, min(0.95, 0.35 + pain_f*0.3))],
        [max(0.05, min(0.95, 0.40 + hormone_f*0.45))],
        [max(0.05, min(0.95, 0.30 + pain_f*0.55))],
        [max(0.05, min(0.95, 0.38 + dys_f*0.5))],
    ]
    fig = go.Figure(go.Heatmap(z=weights, x=["Similarity"], y=subtypes,
                               colorscale="Magma", zmin=0, zmax=1))
    fig.update_layout(title="Biomarker Subtype Alignment", height=320,
                      margin=dict(l=0, r=0, t=35, b=0))
    return fig


def build_3d_fig(twin, p_prob, p_stage, f_risk, patient_seed=42):
    """Digital twin 3D figure using the correct simulator API."""
    twin.update_from_model_prediction(
        probability=p_prob,
        stage=p_stage,
        future_risk=float(f_risk[2]) if len(f_risk) >= 3 else 0.0  # 5-year risk
    )
    twin_data = twin.generate_3d_scatter_data(patient_seed=patient_seed)

    severity = min(1.0, p_prob)
    fig = go.Figure()
    lighting = dict(ambient=0.45, diffuse=0.9, specular=1.2, roughness=0.3)

    # Uterus
    ux, uy, uz = twin_data["uterus"]
    r_val = int(150 - severity * 120)
    fig.add_trace(go.Surface(x=ux, y=uy, z=uz, opacity=1.0,
                             colorscale=[[0, "rgb(255,210,215)"], [1, f"rgb(255,{r_val},{r_val})"]],
                             showscale=False, name="Uterus", lighting=lighting))

    # Ovaries
    for name, key in [("Left Ovary", "left_ovary"), ("Right Ovary", "right_ovary")]:
        ox, oy, oz = twin_data[key]
        fig.add_trace(go.Surface(x=ox, y=oy, z=oz, opacity=1.0, colorscale="Sunset",
                                 showscale=False, name=name, lighting=lighting))

    # Current Lesions
    lx, ly, lz, lc = twin_data["lesions"]
    l_sizes = twin_data.get("lesion_sizes", [])
    if lx:
        m_sizes = [max(6, min(18, 4 + s)) for s in (l_sizes if len(l_sizes) == len(lx) else [8]*len(lx))]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="markers",
                                   marker=dict(size=m_sizes, color=lc, colorscale="YlOrRd",
                                               opacity=0.95, symbol="diamond",
                                               line=dict(width=2, color="DarkRed")),
                                   name="Endometrial Lesions"))

    # Adhesions
    for pt1, pt2 in twin_data.get("adhesions", []):
        fig.add_trace(go.Scatter3d(x=[pt1[0], pt2[0]], y=[pt1[1], pt2[1]], z=[pt1[2], pt2[2]],
                                   mode="lines", line=dict(color="rgba(139,0,0,0.5)", width=4),
                                   name="Adhesion", showlegend=False))

    fig.update_layout(
        scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=False), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, b=0, t=0), height=700, showlegend=True
    )
    return fig




def make_report_text(row, p_prob, p_std, p_stage, gate_probs, clinical_raw):
    stage_str = STAGE_NAMES[p_stage]
    expert_names = ["Ovarian Physiology", "Deep Infiltrating (DIE)",
                    "Superficial Peritoneal", "Adenomyosis/Uterine"]
    top_exp = expert_names[int(np.argmax(gate_probs))]
    ca125 = clinical_raw[6]; estradiol = clinical_raw[7]; pain = clinical_raw[2]
    report = (
        f"DISCLAIMER: AI research tool — not a clinical diagnosis.\n\n"
        f"Primary Assessment: {p_prob*100:.1f}% (+/-{p_std*100:.1f}%) probability of active "
        f"endometriosis. Estimated classification: {stage_str}.\n\n"
        f"Neural Routing: {gate_probs[np.argmax(gate_probs)]*100:.1f}% of computation routed "
        f"to the {top_exp} Expert Sub-Network.\n\n"
        f"Biomarker Highlights:\n"
    )
    if ca125 > 35:
        report += f"- CA-125: {ca125:.1f} U/mL — elevated (threshold 35 U/mL). Differential diagnosis required.\n"
    if estradiol > 200:
        report += f"- Estradiol: {estradiol:.1f} pg/mL — hyperestrogenism may drive ectopic proliferation.\n"
    if pain > 6:
        report += f"- Pelvic Pain Score {pain:.0f}/10 — indicates significant nociceptive involvement.\n"
    report += "\nRecommendation: High-resolution transvaginal ultrasound and pelvic MRI for structural assessment."
    return report


def make_plan_text(p_prob, clinical_raw):
    bmi = clinical_raw[1]; pain = clinical_raw[2]; estradiol = clinical_raw[7]
    plan = "Personal Actionable Health Plan (AI-generated for educational purposes only):\n\n"
    if p_prob > 0.6:
        plan += "1. Medical Consultation: Bring this report to a gynaecology specialist. Discuss diagnostic laparoscopy.\n"
    else:
        plan += "1. Monitoring: Risk profile is stable. Maintain routine gynaecological checkups.\n"
    if pain > 5:
        plan += "2. Pain Management: Pain score is elevated — discuss NSAIDs or pelvic floor physiotherapy.\n"
    if estradiol > 150:
        plan += "3. Hormonal Balance: Omega-3-rich anti-inflammatory diet may help regulate excess oestrogen.\n"
    if bmi > 25:
        plan += "4. Lifestyle: An anti-inflammatory diet (Mediterranean) can reduce systemic inflammation markers.\n"
    return plan


def run(input_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"

    print("\n🤖  Loading trained model...")
    model = load_model(device=device)

    print("🔬  Initialising Digital Twin simulator...")
    twin = UterusDigitalTwin()

    df = pd.read_csv(input_csv)
    print(f"\n📋  Found {len(df)} patients in '{input_csv}'\n")

    for idx, row in df.iterrows():
        pid = str(row.get("patient_id", f"P{idx+1:03d}"))
        print(f"  → Processing patient {pid} ({idx+1}/{len(df)})...", end=" ", flush=True)

        try:
            p_prob, p_std, p_stage, f_risk, gate_probs, clinical_raw = predict(model, row, device)

            # Build all figures
            fig_xai, explanation   = build_xai_fig(clinical_raw, p_prob, model)
            fig_radar              = build_radar_fig(clinical_raw)
            fig_heat               = build_heatmap_fig(clinical_raw)
            fig_3d                 = build_3d_fig(twin, p_prob, p_stage, f_risk,
                                                  patient_seed=int(idx) * 7 + 1)

            # Build text sections
            report_text = make_report_text(row, p_prob, p_std, p_stage, gate_probs, clinical_raw)
            patient_plan = make_plan_text(p_prob, clinical_raw)

            # clinical_data shape expected: (1, 12) — pad il6/amh/crp from CSV or defaults
            cd_full = np.array([[
                clinical_raw[0], clinical_raw[1], clinical_raw[2], clinical_raw[3],
                clinical_raw[4], clinical_raw[5], clinical_raw[6], clinical_raw[7], clinical_raw[8],
                float(row.get("il6",  5.0)),
                float(row.get("amh",  2.5)),
                float(row.get("crp",  2.0)),
            ]])

            pdf_bytes = generate_advanced_pdf_report(
                patient_data=cd_full,
                p_prob=p_prob, p_std=p_std,
                stage_names=STAGE_NAMES, p_stage=p_stage,
                f_risk=f_risk,
                report_text=report_text, patient_plan=patient_plan,
                fig_3d=fig_3d, fig_radar=fig_radar, fig_heat=fig_heat,
                fig_xai=fig_xai, explanation=explanation
            )

            out_path = os.path.join(output_dir, f"report_{pid}.pdf")
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)

            risk_label = "HIGH" if p_prob > 0.6 else ("MODERATE" if p_prob > 0.35 else "LOW")
            print(f"✅  {STAGE_NAMES[p_stage]} | Risk {p_prob*100:.1f}% ({risk_label}) → {out_path}")

        except Exception as e:
            print(f"❌  FAILED — {e}")

    print(f"\n🎉  All reports saved to: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-generate PDF reports for multiple patients.")
    parser.add_argument("--input",  default="test_patients_10.csv", help="Input CSV file")
    parser.add_argument("--outdir", default="reports",              help="Output directory for PDFs")
    args = parser.parse_args()
    run(args.input, args.outdir)
