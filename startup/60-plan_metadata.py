print(ttime.ctime() + ' >>>> ' + __file__)
from xas.file_io import validate_file_exists
from xas.metadata import metadata_dict, ghs_selected_gas_key_match

def populate_standard_metadata_dict(obj_dict=None):
    if obj_dict is None:
        obj_dict = {}

    for key, item in metadata_dict.items():
        if item['kind'] == 'attribute':
            if key in obj_dict.keys():
                item['object'] = obj_dict[key]
            else:
                item['object'] = None
        elif item['kind'] == 'epics_pv':
            if 'pv_str' in item.keys():
                item['object'] = EpicsSignalRO(item['pv_str'], name=key)
            else:

                if key in obj_dict.keys():
                    item['object'] = obj_dict[key]
                else:
                    item['object'] = None

populate_standard_metadata_dict({
'nslsii_status':                  nsls_ii.return_status_string,
'nslsii_current':                 nsls_ii.beam_current,
'nslsii_energy':                  nsls_ii.get_energy_str,
'beamline_cm1':                   cm1.get_current_stripe,
'beamline_cm2':                   cm2.get_current_stripe,
'beamline_cm2_bender':            bender.load_cell,
'beamline_fm':                    fm.get_current_stripe,
'beamline_fm_bender':             bender_fm.load_cell,
'beamline_harmonic_rejection':    hhrm.get_current_stripe,
'angle_offset' :                  hhm.angle_offset,
'angle_offset_deg' :              hhm.get_angle_offset_deg_str,
'mono_encoder_resolution':        hhm.get_mono_encoder_resolution_str,
'detector_i0_n2':                (lambda : f'{np.round(gas_n2.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_it_n2':                (lambda : f'{np.round(gas_n2.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_ir_n2':                (lambda : f'{np.round(gas_n2.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_i0_he':                (lambda : f'{np.round(gas_he.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_it_he':                (lambda : f'{np.round(gas_he.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_ir_he':                (lambda : f'{np.round(gas_he.flow.get() / (gas_he.flow.get() + gas_n2.flow.get()) * 100)}%'),
'detector_i0_volt':              wps1.hv302.read_pv,
'detector_it_volt':              wps1.hv303.read_pv,
'detector_ir_volt':              wps1.hv305.read_pv,
'i0_gain':                       i0_amp.get_gain_value,
'it_gain':                       it_amp.get_gain_value,
'ir_gain':                       ir_amp.get_gain_value,
'iff_gain':                      iff_amp.get_gain_value,
'sample_x_position':             sample_stage.x.user_readback,
'sample_y_position':             sample_stage.y.user_readback,
'sample_z_position':             sample_stage.z.user_readback,
'sample_th_position':            sample_stage.th.user_readback,
'potentiostat_v':                potentiostatV,
'potentiostat_i':                potentiostatI,
})


def get_standard_metadata():
    md = {}
    for key, item in metadata_dict.items():
        if item['kind'] == 'auto':
            continue

        elif item['kind'] == 'attribute':
            object = item['object']
            if object is not None:
                value = item['object']()
            else:
                value = ''

        elif item['kind'] == 'epics_pv':
            object = item['object']
            if object is not None:
                value = item['object'].get()
            else:
                value = ''

        elif item['kind'] == 'fixed_value':
            value = item['value']

        md[key] = value

    return md


def get_general_md():
    md = get_standard_metadata()

    for indx in range(8):
        md[f'ch{indx+1}_offset'] = getattr(apb, f'ch{indx+1}_offset').get()
        _ch = getattr(apb, f'ch{indx+1}')
        amp = _ch.amp
        if amp:
            md[f'ch{indx+1}_amp_gain']= amp.get_gain()[0]
        else:
            md[f'ch{indx+1}_amp_gain'] = 0

    for ghs_gas_key, ghs_gas_match in ghs_selected_gas_key_match.items():
        md[ghs_gas_key] = ghs_gas_match[md[ghs_gas_key]]

    return md



def create_interp_file_name(name, fn_ext):
    fn = f"{ROOT_PATH}/{USER_PATH}/{RE.md['year']}/{RE.md['cycle']}/{RE.md['proposal']}/{name}{fn_ext}"
    fn = validate_file_exists(fn)
    return fn

def get_detector_md(detectors_dict):
    md = {}
    for detector, device_dict in detectors_dict.items():
        device = device_dict['device']
        if hasattr(device, 'read_config_metadata'):
            config = device.read_config_metadata()
        else:
            config = {}
        md[detector] = {'config' : config}
    return md

def get_scan_md(name, comment, detectors_dict, fn_ext):
    fn = create_interp_file_name(name, fn_ext)
    md_general = get_general_md()
    md_detectors = get_detector_md(detectors_dict)
    md_scan = {'interp_filename': fn,
               'name': name,
               'comment': comment,
               'detectors': md_detectors,
               'plot_hint': '$5/$1'}
    return {**md_general, **md_scan}

def get_hhm_scan_md(name, comment, trajectory_filename, detectors_dict, element, e0, edge, metadata, fn_ext='.raw'):
    try:
        full_element_name = getattr(elements, element).name.capitalize()
    except:
        full_element_name = element

    md_scan = get_scan_md(name, comment, detectors_dict, fn_ext)

    md_hhm_scan = {'trajectory_filename': trajectory_filename,
                   'element': element,
                   'element_full': full_element_name,
                   'edge': edge,
                   'e0': e0}
    return {**md_hhm_scan, **md_scan, **metadata}



