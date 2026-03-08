"""
Clinical Input Validation Module
Validates patient biomarker values against physiological ranges to prevent
impossible inputs from reaching the AI inference pipeline.

References:
- CA-125: Mol et al., 1998 (normal < 35 U/mL)
- Estradiol: varies by cycle phase (follicular 30-120, ovulatory 130-370, luteal 70-250 pg/mL)
- Progesterone: follicular < 1.5, luteal 2-25 ng/mL
- IL-6: normal < 7 pg/mL (Harada et al., 2001)
- AMH: age-dependent, 1.0-3.5 ng/mL typical reproductive age
- CRP: normal < 3.0 mg/L
"""

PHYSIOLOGICAL_RANGES = {
    'age': {'min': 10, 'max': 65, 'unit': 'years', 'label': 'Age'},
    'bmi': {'min': 12.0, 'max': 60.0, 'unit': 'kg/m²', 'label': 'BMI'},
    'pelvic_pain': {'min': 0, 'max': 10, 'unit': '/10', 'label': 'Pelvic Pain Score'},
    'dysmenorrhea': {'min': 0, 'max': 10, 'unit': '/10', 'label': 'Dysmenorrhea Score'},
    'ca125': {'min': 0.0, 'max': 1000.0, 'unit': 'U/mL', 'label': 'CA-125'},
    'estradiol': {'min': 0.0, 'max': 2000.0, 'unit': 'pg/mL', 'label': 'Estradiol'},
    'progesterone': {'min': 0.0, 'max': 200.0, 'unit': 'ng/mL', 'label': 'Progesterone'},
    'il6': {'min': 0.0, 'max': 500.0, 'unit': 'pg/mL', 'label': 'IL-6'},
    'amh': {'min': 0.0, 'max': 25.0, 'unit': 'ng/mL', 'label': 'AMH'},
    'crp': {'min': 0.0, 'max': 300.0, 'unit': 'mg/L', 'label': 'CRP'},
}

# Clinical alert thresholds (evidence-based)
BIOMARKER_ALERTS = {
    'ca125': {
        'elevated': 35.0,
        'high': 100.0,
        'message_elevated': 'CA-125 above 35 U/mL is elevated. Note: CA-125 is non-specific and can be elevated in ovarian cancer, PID, pregnancy, and menstruation.',
        'message_high': 'CA-125 above 100 U/mL is significantly elevated. Differential diagnosis required (endometriosis vs. ovarian malignancy).',
    },
    'il6': {
        'elevated': 7.0,
        'high': 50.0,
        'message_elevated': 'IL-6 above 7 pg/mL indicates active systemic inflammation, consistent with endometriosis peritoneal fluid findings.',
        'message_high': 'IL-6 markedly elevated. Rule out concurrent infection or autoimmune conditions.',
    },
    'crp': {
        'elevated': 3.0,
        'high': 10.0,
        'message_elevated': 'CRP above 3.0 mg/L indicates low-grade systemic inflammation.',
        'message_high': 'CRP above 10.0 mg/L indicates significant inflammation. Consider concurrent infection.',
    },
    'estradiol': {
        'elevated': 300.0,
        'high': 500.0,
        'message_elevated': 'Estradiol above 300 pg/mL may indicate hyperestrogenism, a key driver of ectopic endometrial proliferation.',
        'message_high': 'Estradiol markedly elevated. Consider ovarian hyperstimulation or estrogen-secreting pathology.',
    },
}

# Cycle-phase reference ranges for hormone contextualization
CYCLE_PHASE_RANGES = {
    'Follicular (Day 1-13)': {
        'estradiol': (30, 120),
        'progesterone': (0.1, 1.5),
        'description': 'Early cycle. Estradiol rises gradually as follicles develop.',
    },
    'Ovulatory (Day 14)': {
        'estradiol': (130, 370),
        'progesterone': (0.5, 2.0),
        'description': 'Mid-cycle surge. Estradiol peaks triggering LH surge.',
    },
    'Luteal (Day 15-28)': {
        'estradiol': (70, 250),
        'progesterone': (2.0, 25.0),
        'description': 'Post-ovulation. Progesterone dominant from corpus luteum.',
    },
    'Unknown / Not Specified': {
        'estradiol': (30, 400),
        'progesterone': (0.1, 25.0),
        'description': 'Cycle phase unknown. Using full physiological range.',
    },
}


def validate_clinical_input(data_dict):
    """
    Validates clinical input values against physiological ranges.
    
    Args:
        data_dict: dict with keys matching PHYSIOLOGICAL_RANGES keys and numeric values
        
    Returns:
        tuple: (is_valid: bool, warnings: list[str], errors: list[str])
    """
    warnings = []
    errors = []
    
    for key, value in data_dict.items():
        if key not in PHYSIOLOGICAL_RANGES:
            continue
            
        bounds = PHYSIOLOGICAL_RANGES[key]
        label = bounds['label']
        unit = bounds['unit']
        
        if value < bounds['min']:
            errors.append(f"{label}: {value} {unit} is below physiological minimum ({bounds['min']} {unit})")
        elif value > bounds['max']:
            errors.append(f"{label}: {value} {unit} exceeds physiological maximum ({bounds['max']} {unit})")
    
    # Check biomarker alert thresholds
    for key, thresholds in BIOMARKER_ALERTS.items():
        if key in data_dict:
            value = data_dict[key]
            if value >= thresholds['high']:
                warnings.append(f"⚠️ {thresholds['message_high']}")
            elif value >= thresholds['elevated']:
                warnings.append(f"📊 {thresholds['message_elevated']}")
    
    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def get_cycle_context(cycle_phase, estradiol_val, progesterone_val):
    """
    Contextualizes hormone values based on menstrual cycle phase.
    
    Returns:
        str: Clinical context message about the hormone levels relative to cycle phase.
    """
    if cycle_phase not in CYCLE_PHASE_RANGES:
        cycle_phase = 'Unknown / Not Specified'
    
    ranges = CYCLE_PHASE_RANGES[cycle_phase]
    messages = []
    
    e_min, e_max = ranges['estradiol']
    if estradiol_val < e_min:
        messages.append(f"Estradiol ({estradiol_val:.1f} pg/mL) is LOW for {cycle_phase} phase (expected {e_min}-{e_max} pg/mL)")
    elif estradiol_val > e_max:
        messages.append(f"Estradiol ({estradiol_val:.1f} pg/mL) is HIGH for {cycle_phase} phase (expected {e_min}-{e_max} pg/mL)")
    else:
        messages.append(f"Estradiol ({estradiol_val:.1f} pg/mL) is within normal range for {cycle_phase} phase")
    
    p_min, p_max = ranges['progesterone']
    if progesterone_val < p_min:
        messages.append(f"Progesterone ({progesterone_val:.1f} ng/mL) is LOW for {cycle_phase} phase (expected {p_min}-{p_max} ng/mL)")
    elif progesterone_val > p_max:
        messages.append(f"Progesterone ({progesterone_val:.1f} ng/mL) is HIGH for {cycle_phase} phase (expected {p_min}-{p_max} ng/mL)")
    else:
        messages.append(f"Progesterone ({progesterone_val:.1f} ng/mL) is within normal range for {cycle_phase} phase")
    
    return messages
