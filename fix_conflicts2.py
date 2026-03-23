with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
state = 'NORMAL'
head_lines = []
tail_lines = []

for line in lines:
    if line.startswith('<<<<<<< HEAD'):
        state = 'HEAD'
        head_lines = []
        tail_lines = []
    elif line.startswith('======='):
        if state == 'HEAD':
            state = 'TAIL'
        else:
            out.append(line)
    elif line.startswith('>>>>>>>'):
        if state == 'TAIL':
            state = 'NORMAL'
            head_text = "".join(head_lines)
            tail_text = "".join(tail_lines)
            
            if "from digital_twin.mesh_loader" in head_text:
                out.append(head_text)
                out.append(tail_text)
            elif "page.extract_text" in head_text and "mock_ai_extract_to_df" in head_text:
                fixed_head = head_text.replace("elif file_ext in _IMAGE_EXTENSIONS:", "elif file_ext in ['png', 'jpg', 'jpeg'] or file_ext in getattr(globals(), '_IMAGE_EXTENSIONS', ['png', 'jpg', 'jpeg']):")
                out.append(fixed_head)
            elif "Sensor / wearable report detected" in head_text:
                out.append(head_text)
            elif "clinical_data = np.array" in head_text and "should_run" in head_text:
                out.append('''            st.subheader("Advanced Inflammatory Markers")
            il6 = st.slider("IL-6 (pg/mL)", 0.0, 500.0, min(500.0, max(0.0, float(default_vals.get('il6', 15.0)))))
            amh = st.slider("AMH (ng/mL)", 0.0, 25.0, min(25.0, max(0.0, float(default_vals.get('amh', 2.5)))))
            crp = st.slider("CRP (mg/L)", 0.0, 300.0, min(300.0, max(0.0, float(default_vals.get('crp', 12.0)))))
            
            # Show cycle-phase hormone context
            try:
                from services.clinical_validator import get_cycle_context
                cycle_msgs = get_cycle_context(cycle_phase, estradiol, prog)
                with st.expander("🔬 Hormone Cycle Context", expanded=False):
                    for msg in cycle_msgs:
                        st.info(msg)
            except Exception:
                pass
            
            clinical_data = np.array([[age, bmi, pelvic_pain, dysmenorrhea, dyspareunia, fam_hx, ca125, estradiol, prog, il6, amh, crp]])
            
            run_prediction = st.button("▶️ Run prediction", type="primary", help="Compute risk score and XAI attribution. Click after adjusting parameters.")
            st.caption("✨ Adjust parameters above, then click Run prediction to compute results without freezing UI.")

        with col_results:
            should_run = run_prediction or not st.session_state.get('prediction_computed', False)
            last_computed = st.session_state.get('last_computed_clinical_data')
            params_changed = last_computed is not None and not np.allclose(clinical_data, last_computed, rtol=1e-5)
            if params_changed and st.session_state.get('prediction_computed'):
                st.info("📝 Parameters changed. Click **Run prediction** to update results.")
            if should_run:
                with st.spinner("Computing prediction..."):
                    mock_means = np.array([32.0, 25.0, 5.0, 5.0, 0.5, 0.5, 45.0, 150.0, 10.0, 15.0, 2.5, 12.0])
                    mock_stds = np.array([7.0, 4.0, 3.0, 3.0, 0.5, 0.5, 15.0, 50.0, 5.0, 5.0, 1.0, 5.0])
                    tensor_data = torch.tensor((clinical_data - mock_means) / mock_stds, dtype=torch.float32)
                    if st.session_state.get('us_embedding_from_image') is not None:
                        emb = st.session_state['us_embedding_from_image']
                        us_data = torch.tensor(emb, dtype=torch.float32) if isinstance(emb, np.ndarray) and emb.shape == (1, 128) else torch.zeros((1, 128), dtype=torch.float32)
                    else:
                        us_data = torch.zeros((1, 128), dtype=torch.float32)
                    genomic_data = torch.zeros((1, 256), dtype=torch.float32)
                    path_data = torch.zeros((1, 64), dtype=torch.float32)
                    sensor_data = torch.zeros((1, 32), dtype=torch.float32)
                    with torch.no_grad():
                        prob, stage_logits, future_risk, gate_probs = model(tensor_data, us_data, genomic_data, path_data, sensor_data)
                        model.eval()
                        for m in model.modules():
                            if m.__class__.__name__.startswith('Dropout'):
                                m.train()
                        
                        mc_samples = 5
                        tensor_data_mc = tensor_data.repeat(mc_samples, 1)
                        us_data_mc = us_data.repeat(mc_samples, 1)
                        genomic_data_mc = genomic_data.repeat(mc_samples, 1)
                        path_data_mc = path_data.repeat(mc_samples, 1)
                        sensor_data_mc = sensor_data.repeat(mc_samples, 1)
                        p_mc, _, _, _ = model(tensor_data_mc, us_data_mc, genomic_data_mc, path_data_mc, sensor_data_mc)
                        mc_probs = p_mc.squeeze().tolist()
                        mean_p = np.mean(mc_probs)
                        std_p = np.std(mc_probs)
                        
                        for m in model.modules():
                            if m.__class__.__name__.startswith('Dropout'):
                                m.eval()
                        
                        if not os.path.exists('global_model.pth'):
                            demo_risk = (pelvic_pain + dysmenorrhea) / 20.0 * 0.4 + (ca125 / 150.0) * 0.3 + (estradiol / 500.0) * 0.3
                            demo_risk = min(0.99, max(0.01, demo_risk))
                            prob = torch.tensor([[demo_risk]])
                            stage_idx = min(4, int(demo_risk * 5))
                            stage_logits = torch.zeros((1, 5))
                            stage_logits[0, stage_idx] = 10.0
                            f1, f3, f5 = min(0.99, demo_risk * 1.1), min(0.99, demo_risk * 1.3), min(0.99, demo_risk * 1.6)
                            future_risk = torch.tensor([[f1, f3, f5]])
                            gate_probs = torch.tensor([[0.05, 0.8, 0.1, 0.05]]) if pelvic_pain > 7 else torch.tensor([[0.5, 0.1, 0.3, 0.1]])
                            mean_p, std_p = prob.item(), 0.02 + (demo_risk * 0.05)
                        else:
                            prob = torch.tensor([[mean_p]])
                    st.session_state['pred_prob'] = prob.item()
                    st.session_state['pred_prob_std'] = std_p
                    st.session_state['pred_stage'] = torch.argmax(stage_logits, dim=1).item()
                    st.session_state['future_risk'] = future_risk.squeeze().numpy()
                    st.session_state['gate_probs'] = gate_probs.squeeze().numpy()
                    st.session_state['clinical_data'] = clinical_data
                    st.session_state['prediction_computed'] = True
                    st.session_state['last_computed_clinical_data'] = clinical_data.copy()
''')
            elif "f_risk_val" in head_text and "_cached_twin_data" in head_text:
                head_text = head_text.replace("res = 0.5 if low_detail else 1.0", "res = 0.2 if low_detail else 0.4")
                out.append(head_text)
            else:
                out.append(head_text)
        else:
            out.append(line)
    else:
        if state == 'NORMAL':
            out.append(line)
        elif state == 'HEAD':
            head_lines.append(line)
        elif state == 'TAIL':
            tail_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(out)

print(f"Merge successful. Wrote {len(out)} lines to app.py")
