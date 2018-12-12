prepare_bl_def = [
    [
        {
            "energy_start": 0,
            "energy_end": 6000,
            "pvs": {
                "IC Gas He": {
                    "name": gas_he.flow.name,
                    "object": gas_he.flow,
                    "value": 5
                },
                "IC Gas N2": {
                    "name": gas_n2.flow.name,
                    "object": gas_n2.flow,
                    "value": 1
                },
                "I0 Voltage": {
                    "name": wps1.hv302.name,
                    "object": wps1.hv302,
                    "value": 800
                },
                "It Voltage": {
                    "name": wps1.hv303.name,
                    "object": wps1.hv303,
                    "value": 800
                },
                "Ir Voltage": {
                    "name": wps1.hv305.name,
                    "object": wps1.hv305,
                    "value": 800
                },
                "Filterbox Pos": {
                    "name": filterbox.y.name,
                    "object": filterbox.y,
                    "STS PVS": [filterbox.pos1,
                                filterbox.pos2,
                                filterbox.pos3,
                                filterbox.pos4,
                                filterbox.pos5],
                    "value": 1,
                },
                "HHRM Hor Trans": {
                    "name": hhrm.hor_translation.name,
                    "object": hhrm.hor_translation,
                    "value": 0
                },
                "BPMs": [
                    {
                        "name": bpm_cm.name,
                        "object": bpm_cm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_fm.name,
                        "object": bpm_fm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt1.name,
                        "object": bpm_bt1,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt2.name,
                        "object": bpm_bt2,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_es.name,
                        "object": bpm_es,
                        "exposure_time": -1,
                        "value": "IN"
                    }
                ]
            }
        },
        {
            "energy_start": 6000,
            "energy_end": 10000,
            "pvs": {
                "IC Gas He": {
                    "name": gas_he.flow.name,
                    "object": gas_he.flow,
                    "value": 5
                },
                "IC Gas N2": {
                    "name": gas_n2.flow.name,
                    "object": gas_n2.flow,
                    "value": 3
                },
                "I0 Voltage": {
                    "name": wps1.hv302.name,
                    "object": wps1.hv302,
                    "value": 1100
                },
                "It Voltage": {
                    "name": wps1.hv303.name,
                    "object": wps1.hv303,
                    "value": 1100
                },
                "Ir Voltage": {
                    "name": wps1.hv305.name,
                    "object": wps1.hv305,
                    "value": 1100
                },
                "Filterbox Pos": {
                    "name": filterbox.y.name,
                    "object": filterbox.y,
                    "STS PVS": [filterbox.pos1,
                                filterbox.pos2,
                                filterbox.pos3,
                                filterbox.pos4,
                                filterbox.pos5],
                    "value": 2,
                },
                "HHRM Hor Trans": {
                    "name": hhrm.hor_translation.name,
                    "object": hhrm.hor_translation,
                    "value": 0
                },
                "BPMs": [
                    {
                        "name": bpm_cm.name,
                        "object": bpm_cm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_fm.name,
                        "object": bpm_fm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt1.name,
                        "object": bpm_bt1,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt2.name,
                        "object": bpm_bt2,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_es.name,
                        "object": bpm_es,
                        "exposure_time": -1,
                        "value": "IN"
                    }
                ]
            }
        },
        {
            "energy_start": 10000,
            "energy_end": 17000,
            "pvs": {
                "IC Gas He": {
                    "name": gas_he.flow.name,
                    "object": gas_he.flow,
                    "value": 5
                },
                "IC Gas N2": {
                    "name": gas_n2.flow.name,
                    "object": gas_n2.flow,
                    "value": 5
                },
                "I0 Voltage": {
                    "name": wps1.hv302.name,
                    "object": wps1.hv302,
                    "value": 1700
                },
                "It Voltage": {
                    "name": wps1.hv303.name,
                    "object": wps1.hv303,
                    "value": 1700
                },
                "Ir Voltage": {
                    "name": wps1.hv305.name,
                    "object": wps1.hv305,
                    "value": 1700
                },
                "Filterbox Pos": {
                    "name": filterbox.y.name,
                    "object": filterbox.y,
                    "STS PVS": [filterbox.pos1,
                                filterbox.pos2,
                                filterbox.pos3,
                                filterbox.pos4,
                                filterbox.pos5],
                    "value": 3,
                },
                "HHRM Hor Trans": {
                    "name": hhrm.hor_translation.name,
                    "object": hhrm.hor_translation,
                    "value": 80
                },
                "BPMs": [
                    {
                        "name": bpm_cm.name,
                        "object": bpm_cm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_fm.name,
                        "object": bpm_fm,
                        "exposure_time": -1,
                        "value": "IN"
                    },
                    {
                        "name": bpm_bt1.name,
                        "object": bpm_bt1,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt2.name,
                        "object": bpm_bt2,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_es.name,
                        "object": bpm_es,
                        "exposure_time": -1,
                        "value": "IN"
                    }
                ]
            }
        },
        {
            "energy_start": 17000,
            "energy_end": 50000,
            "pvs": {
                "IC Gas He": {
                    "name": gas_he.flow.name,
                    "object": gas_he.flow,
                    "value": 2
                },
                "IC Gas N2": {
                    "name": gas_n2.flow.name,
                    "object": gas_n2.flow,
                    "value": 5
                },
                "I0 Voltage": {
                    "name": wps1.hv302.name,
                    "object": wps1.hv302,
                    "value": 1700
                },
                "It Voltage": {
                    "name": wps1.hv303.name,
                    "object": wps1.hv303,
                    "value": 1700
                },
                "Ir Voltage": {
                    "name": wps1.hv305.name,
                    "object": wps1.hv305,
                    "value": 1700
                },
                "Filterbox Pos": {
                    "name": filterbox.y.name,
                    "object": filterbox.y,
                    "STS PVS": [filterbox.pos1,
                                filterbox.pos2,
                                filterbox.pos3,
                                filterbox.pos4,
                                filterbox.pos5],
                    "value": 4,
                },
                "HHRM Hor Trans": {
                    "name": hhrm.hor_translation.name,
                    "object": hhrm.hor_translation,
                    "value": 80
                },
                "BPMs": [
                    {
                        "name": bpm_cm.name,
                        "object": bpm_cm,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_fm.name,
                        "object": bpm_fm,
                        "exposure_time": -1,
                        "value": "IN"
                    },
                    {
                        "name": bpm_bt1.name,
                        "object": bpm_bt1,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_bt2.name,
                        "object": bpm_bt2,
                        "exposure_time": -1,
                        "value": "OUT"
                    },
                    {
                        "name": bpm_es.name,
                        "object": bpm_es,
                        "exposure_time": -1,
                        "value": "IN"
                    }
                ]
            }
        }
    ],
    {
        "FB Positions": [
            1,
            -69,
            -139,
            -209,
            -279
        ]
    }
]