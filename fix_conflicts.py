import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Block 1: Lines ~879 to ~952 (PDF extraction)
# I will use a regex to capture this block and replace it with a clean version.
pattern1 = r"<<<<<<< HEAD\n\s*text = \" \"\.join\(\[page\.extract_text\(\).*?>>>>>>> [a-z0-9]+"

replacement1 = """                        text = " ".join([page.extract_text() or "" for page in reader.pages])
                        if not text.strip():
                            st.warning("PDF has no extractable text (e.g. scanned). Upload an image of the report for OCR.")
                            df = None
                        else:
                            with st.spinner("🤖 AI Extracting data from PDF..."):
                                df = mock_ai_extract_to_df(text)
                    elif file_ext in ['png', 'jpg', 'jpeg']:
                        from PIL import Image
                        import pytesseract
                        try:
                            uploaded_file.seek(0)
                            image = Image.open(uploaded_file).copy()
                            if image.mode not in ('RGB', 'L'):
                                image = image.convert('RGB')
                            text = pytesseract.image_to_string(image)
                            if not text.strip():
                                st.warning("No text detected in image. Using default parameters.")
                                df = pd.DataFrame([{'age': 32, 'bmi': 24.5, 'pelvic_pain_score': 8, 'dysmenorrhea_score': 7, 'ca125': 65.0, 'estradiol': 250.0}])
                            else:
                                with st.spinner("🤖 AI Extracting data from Image..."):
                                    df = mock_ai_extract_to_df(text)
                        except Exception as img_err:
                            st.warning(f"Image/OCR error: {img_err}. Using default parameters.")
                            df = pd.DataFrame([{'age': 32, 'bmi': 24.5, 'pelvic_pain_score': 8, 'dysmenorrhea_score': 7, 'ca125': 65.0, 'estradiol': 250.0}])"""

content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# Block 2: Lines ~1013 to ~1191 (Predictor)
# We will use regex to find this entire block. We keep the manual run button but incorporate IL6, AMH, CRP.
# We have to be careful not to match too much.
pattern2 = r"<<<<<<< HEAD\n\s*clinical_data = np\.array\(\[\[age, bmi, pelvic_pain.*?>>>>>>> [a-z0-9]+"

replacement2 = """            st.subheader("Advanced Inflammatory Markers")
            il6 = st.slider("IL-6 (pg/mL)", 0.0, 500.0, min(500.0, max(0.0, float(default_vals.get('il6', 15.0)))))
            amh = st.slider("AMH (ng/mL)", 0.0, 25.0, min(25.0, max(0.0, float(default_vals.get('amh', 2.5)))))
            crp = st.slider("CRP (mg/L)", 0.0, 300.0, min(300.0, max(0.0, float(default_vals.get('crp', 12.0)))))
            
            # Validate inputs
            input_dict = {'age': age, 'bmi': bmi, 'pelvic_pain': pelvic_pain, 'dysmenorrhea': dysmenorrhea,
                          'ca125': ca125, 'estradiol': estradiol, 'progesterone': prog, 'il6': il6, 'amh': amh, 'crp': crp}
            
            try:
                from services.clinical_validator import validate_clinical_input, get_cycle_context
                is_valid, val_warnings, val_errors = validate_clinical_input(input_dict)
                if val_errors:
                    for err in val_errors:
                        st.error(f"⛔ {err}")
                if val_warnings:
                    with st.expander("📊 Biomarker Alerts (click to expand)", expanded=False):
                        for warn in val_warnings:
                            st.warning(warn)
                # Show cycle-phase hormone context
                cycle_msgs = get_cycle_context(cycle_phase, estradiol, prog)
                with st.expander("🔬 Hormone Cycle Context", expanded=False):
                    for msg in cycle_msgs:
                        st.info(msg)
            except ImportError:
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
                    # Pad inputs to 12 if model expects 12
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
                        # Batch MC Dropout optimization
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
                    st.session_state['last_computed_clinical_data'] = clinical_data.copy()"""

content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Substituted patterns. Length of content is now {len(content)}")
