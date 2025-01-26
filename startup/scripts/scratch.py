import numpy as np
import pandas as pd

centroid_y = EpicsSignal('XF:08IDB-BI{BPM:SP-2}Stats1:CentroidY_RBV', name='centroid_y')
intensity = EpicsSignal('XF:08IDB-BI{BPM:SP-2}Stats1:Total_RBV', name='intensity')

centroid_y_data = {'value': [], 'timestamp': []}
intensity_data =  {'value': [], 'timestamp': []}

def cb_add_data(value, **kwargs):
    # print_to_gui(f"{value = }", add_timestamp=True)
    if len(centroid_y_data) >=10000:
        centroid_y_data['value'].pop(0)
        centroid_y_data['timestamp'].pop(0)
    centroid_y_data['value'].append(value)
    centroid_y_data['timestamp'].append(kwargs['timestamp'])

centroid_y.subscribe(cb_add_data)

def cb_add_data_int(value, **kwargs):
    # print_to_gui(f"{value = }", add_timestamp=True)
    if len(intensity_data) >=10000:
        intensity_data['value'].pop(0)
        intensity_data['timestamp'].pop(0)
    intensity_data['value'].append(value)
    intensity_data['timestamp'].append(kwargs['timestamp'])

intensity.subscribe(cb_add_data_int)


def get_fft(t, s):
    freq = np.fft.fftfreq(t.size, d=t[1] - t[0])
    s_fft = np.abs(np.fft.fft(s))[np.argsort(freq)]
    freq = freq[np.argsort(freq)]
    return freq[freq>0], s_fft[freq>0]

def plot_fft(t, s, *args, **kwargs):
    f, s_fft = get_fft(np.array(t), np.array(s))
    plt.plot(f, s_fft, *args, **kwargs)

def plot_trace_info(data_dict, fignum=1, clear=True):
    t = np.linspace(data_dict['timestamp'][0], data_dict['timestamp'][-1], len(data_dict['timestamp']))
    v = np.interp(t, data_dict['timestamp'], data_dict['value'])

    # t = np.linspace(intensity_data['timestamp'][0], intensity_data['timestamp'][-1], len(intensity_data['timestamp']))
    # v = np.interp(t, intensity_data['timestamp'], intensity_data['value'])

    freq, centroid_fft = get_fft(t, v)
    # freq = np.fft.fftfreq(v.size, d=t[1] - t[0])
    # centroid_fft = np.abs(np.fft.fft(v))[np.argsort(freq)]
    # freq = freq[np.argsort(freq)]

    plt.figure(fignum, clear=clear)
    plt.subplot(221)
    plt.plot(data_dict['timestamp'], data_dict['value'] - np.mean(data_dict['value']))

    plt.subplot(222)
    plt.hist(data_dict['value'] - np.mean(data_dict['value']), 100, density=True);

    plt.subplot(223)
    plt.semilogy(freq, centroid_fft)
    plt.xlim(0.1, 15)
    # plt.ylim(1e2, 1e6)
# plt.semilogy(centroid_fft)


plot_trace_info(centroid_y_data, )
plot_trace_info(intensity_data, fignum=2)

centroid_y_data_85_3 = copy.deepcopy(centroid_y_data)
intensity_data_85_3 = copy.deepcopy(intensity_data)

plot_trace_info(centroid_y_data_85, clear=True, fignum=1)
plot_trace_info(centroid_y_data_65, clear=False, fignum=1)

plot_trace_info(intensity_data_85, fignum=2)

plot_trace_info(centroid_y_data_85_2, clear=True, fignum=1)
plot_trace_info(centroid_y_data_85_3, clear=False, fignum=1)

plot_trace_info(intensity_data_85_2, clear=True, fignum=2)
plot_trace_info(intensity_data_85_3, clear=False, fignum=2)

###

centroid_y_340 = {}
centroid_y_340['value'] = np.array(centroid_y_data['value'])[np.array(centroid_y_data['timestamp'])<=intensity_data['timestamp'][1400]]
centroid_y_340['timestamp'] = np.array(centroid_y_data['timestamp'])[np.array(centroid_y_data['timestamp'])<=intensity_data['timestamp'][1400]]

centroid_y_225 = {}
centroid_y_225['value'] = np.array(centroid_y_data['value'])[(np.array(centroid_y_data['timestamp'])>=intensity_data['timestamp'][28904]) &
                                                             (np.array(centroid_y_data['timestamp']) <= intensity_data['timestamp'][43600])]
centroid_y_225['timestamp'] = np.array(centroid_y_data['timestamp'])[(np.array(centroid_y_data['timestamp'])>=intensity_data['timestamp'][28904]) &
                                                             (np.array(centroid_y_data['timestamp']) <= intensity_data['timestamp'][43600])]

plot_trace_info(centroid_y_340 )
plot_trace_info(centroid_y_225 )

plt.figure(3, clear=True)
plt.hist(centroid_y_340['value'] - np.mean(centroid_y_340['value']), 100, alpha=0.3);
plt.hist(centroid_y_225['value'] - np.mean(centroid_y_225['value']), 100, alpha=0.3);


bla = np.array(intensity_data['value'])[(np.array(intensity_data['timestamp'])>=intensity_data['timestamp'][28904]) &
                                                             (np.array(intensity_data['timestamp']) <= intensity_data['timestamp'][43600])]
plt.figure(); plt.plot(bla)


apb_ave.set_exposure_time(10)
RE(bp.count([apb_ave]))

t = db[-1].table()

# gain was set to 5
t_all_connected_no_scope = db['af703640-8980-4dcd-9545-4c8f0fce4046'].table()
t_all_connected_no_scope_ir_it = db['4d156431-bb99-47e0-af71-c1f1b8b3c0ec'].table()

t_no_amplifier_no_scope = db['ced4d069-6d30-443f-913c-704afa08b5a2'].table()
t_all_conneced_no_scope_with_dominik = db['1c8eaaf9-8d6a-4b86-8255-8559be5d9591'].table()

t_all_connected_no_scope_manual_gain = db['abc456fc-2014-4fab-b55c-6420c7a11e79'].table()
t_all_connected_no_scope_manual_gain_10MHz = db['676abda3-cb15-4612-a6a7-3c88d3be874e'].table()
t_all_connected_no_scope_manual_gain_FB = db['d44ce8d5-0fb3-4af6-985e-346dc98c3d93'].table()
t_all_connected_no_scope_direct_cable = db['aa5c3275-e63f-453a-847d-3d54cce0b015'].table()
t_all_connected_no_scope_direct_cable_with_grounding = db['6869fdc4-1fc9-4057-88ed-53c636bfed4a'].table()

# t_all_connected_no_scope_direct_cable_keithley = db['e0e2ef0e-796a-49d3-aad4-f8929762fe12'].table()
t_all_connected_no_scope_direct_cable_keithley = db['0d265f78-2950-4035-a5b8-60a148af1f19'].table()

t_all_connected_no_scope_direct_cable_ir_held_in_air = db['51f7ed41-e51d-4a5c-a45f-6f1db556ddd1'].table()

t_hv_on = db['62165c00-4b05-4883-850c-fa1510fcad8d'].table()
t_i0_hv_off = db['ae9987b5-7abf-4802-bf72-9067047480ea'].table()
t_all_hv_off = db['964d21f1-0d25-4ffa-91f7-10bd2c5e29b8'].table()

t_apb_only = db['a363016d-8f62-475b-b95c-20ac01172f91'].table()

def _plot_traces(_time, _v, *args, **kwargs):
    plt.subplot(221)
    plt.plot(_time, _v - np.mean(_v), *args, **kwargs)

    plt.subplot(222)
    plt.hist(_v - np.mean(_v), 50, alpha=0.5);

    plt.subplot(223)
    plot_fft(_time, _v, *args, **kwargs)

def plot_traces(t, *args, ch=1, **kwargs):
    _time = np.array(t[f'apb_ave_time_wf'][1])
    _v = np.array(t[f'apb_ave_ch{ch}_wf'][1])
    _plot_traces(_time, _v, *args, **kwargs)


plt.figure()
# plot_traces(t_hv_on, label='HV ON')
# plot_traces(t_i0_hv_off, label='I0 HV OFF')
# plot_traces(t_all_hv_off, label='ALL HV OFF')
plot_traces(t_apb_only, ch=1, label='APB only I0')
plot_traces(t_apb_only, ch=2, label='APB only It')
# plot_traces(t_apb_only, ch=3, label='APB only Ir')
# plot_traces(t_apb_only, ch=4, label='APB only If')



plt.legend()
plt.yscale('log')


bla1 = np.array(t_hv_on['apb_ave_ch1_wf'][1])
bla2 = np.array(t_hv_on['apb_ave_ch2_wf'][1])
bla3 = np.array(t_hv_on['apb_ave_ch3_wf'][1])
bla4 = np.array(t_hv_on['apb_ave_ch4_wf'][1])

bla = np.vstack((bla1, bla2, bla3, bla4))

plt.figure()
plt.plot(bla2, bla3, 'k.')
plt.axis('square')

plt.figure()
plt.imshow(np.corrcoef(bla))
# plot_traces(t_all_connected_no_scope, label='no scope')
# plot_traces(t_all_connected_no_scope_direct_cable, label='FEMTO direct cable no scope')
# plot_traces(t_all_connected_no_scope_direct_cable_keithley, label='KEITHLEY direct cable no scope')

# plot_traces(t_all_connected_no_scope_direct_cable_with_grounding, label='direct cable and grounding no scope')

# plot_traces(t_all_conneced_no_scope_with_dominik, label='dominik no scope')
# plot_traces(t_all_connected_no_scope_manual_gain, label='1 MHzmanual gain no scope')
# plot_traces(t_all_connected_no_scope_manual_gain_10MHz, label='10 Mhz manual gain no scope')
# plot_traces(t_all_connected_no_scope_manual_gain_FB, label='FB manual gain no scope')
plot_traces(t_no_amplifier_no_scope, label='no amplifier no scope')

plt.legend()
plt.yscale('log')



plt.figure()
plot_traces(t_all_connected_no_scope_ir_it, label='no scope IR', ch=3)
plot_traces(t_all_connected_no_scope_direct_cable_ir_held_in_air, label='no scope IR in air', ch=3)

# plot_traces(t_all_connected_no_scope_ir_it, label='with scope Ir', ch=2)
# plot_traces(t_all_connected_no_scope_ir_it, label='with scope It', ch=3)
# plot_traces(t_all_connected_no_scope_ir_it, label='no scope Ir', ch=3)
# plot_traces(t_all_conneced_no_scope_with_dominik, label='dominik no scope')
# plot_traces(t_all_connected_no_scope_manual_gain, label='manual gain no scope')
# plot_traces(t_all_connected_no_scope_manual_gain_10MHz, label='10 Mhz no scope')
# plot_traces(t_all_connected_no_scope_manual_gain_FB, label='FB no scope')
# plot_traces(t_no_amplifier_no_scope, label='no amplifier no scope')

plt.figure()
_plot_traces(t_all_connected_no_scope[f'apb_ave_time_wf'][1],
             np.array(t_all_connected_no_scope['apb_ave_ch2_wf'][1]) / np.array(t_all_connected_no_scope['apb_ave_ch1_wf'][1]))



RE(bp.count([apb_ave]))


t_85 = db['668a4f77-567e-4516-ba8e-a861740298ae'].table()
t_65 = db['d69716cf-454b-4f9d-b567-d5f9ea18b6be'].table()

# no expansion tank
t_85_2 = db['d8828258-96d6-4b0e-96d1-da68c099afb7'].table()

# with expansion tank
t_85_3 = db['b33a284e-a12f-4762-a57d-9a280e874b0f'].table()

t_5000 = db['280aec42-a42c-44fd-9862-bbc3f7339d65'].table()

t_2021 = db['65eb5b76-4598-43f0-bde0-233df7a3a5db'].table()

plt.figure()
# plt.plot(t_2021.apb_ave_time_wf[1], np.array(t_2021.apb_ave_ch1_wf[1])/ 1560 * 930)
# plt.plot(t_5000.apb_ave_time_wf[1], t_5000.apb_ave_ch1_wf[1])
# plt.plot(t_85.apb_ave_time_wf[1], t_85.apb_ave_ch1_wf[1])
# plt.plot(t_65.apb_ave_time_wf[1], t_65.apb_ave_ch1_wf[1])

# plot_traces(t_85, label='85 Hz')
plot_traces(t_85_2, label='85 Hz no expansion tank')
plot_traces(t_85_3, label='85 Hz with expansion tank')
# plot_traces(t_65, label='65 Hz')
plt.legend()
plt.yscale('log')

# plt.hist(t_2021.apb_ave_ch1_wf[1]/np.mean(t_2021.apb_ave_ch1_wf[1]), 100, density=True, alpha=0.5);
# plt.hist(t_85.apb_ave_ch1_wf[1]/np.mean(t_85.apb_ave_ch1_wf[1]), 100, density=True, alpha=0.5);
# plt.xlabel('intensity')
# plt.ylabel('')

hdr = db[-1]
_t = hdr.table(stream_name='apb_stream', fill=True)['apb_stream'][1]
t_apb = hdr.table(stream_name='apb_stream', fill=True)['apb_stream'][1]
# _t = hdr.table(stream_name='pb9_enc1', fill=True)['pb9_enc1'][1]
# t_enc = hdr.table(stream_name='pb9_enc1', fill=True)['pb9_enc1'][1]

# plt.figure(); plt.plot(t_enc[:, 0] + t_enc[:, 1]*1e-9, t_enc[:, 2])
plt.plot(t_apb[:, 0], t_apb[:, 1])



img = bpm_es.get_image_array_data_reshaped()




# dark_image = picam.image.array_data.get().reshape(2048, 2048)
xray_image = picam.image.array_data.get().reshape(2048, 2048)


VMIN, VMAX = 0, 6000
plt.figure(1, clear=True)
plt.subplot(131)
plt.imshow(dark_image, vmin=VMIN, vmax=VMAX)
plt.title(f'Dark ({VMIN}-{VMAX})')

plt.subplot(132)
plt.imshow(xray_image, vmin=VMIN, vmax=VMAX)
plt.title(f'Bright ({VMIN}-{VMAX})')

plt.subplot(133)
# plt.imshow(xray_image - dark_image, vmin=VMIN, vmax=VMAX)
plt.imshow(xray_image - dark_image, vmin=0, vmax=100)
plt.title(f'Difference ({VMIN}-{VMAX})')


##
t = db[-1].table()
energy = t.hhm_energy.values
intensity = t.picam_stats1_total.values
intensity = normalize_peak(intensity)
Ecen0, fwhm0 = estimate_center_and_width_of_peak(energy, intensity)
Ecen, fwhm, intensity_cor, intensity_fit, intensity_fit_raw = fit_gaussian(energy, intensity, Ecen0, fwhm0)


##########

RE(bp.count([picam], 6))

uid_xray = '95e43edc-9507-44f1-966f-db923b7338f4'
uid_dark = 'fc298c99-fd12-4ff6-b915-294e9efadf91'


def _get_images(uids):
    if type(uids) == str:
        uids = [uids]

    imgs = []
    for uid in uids:
        t = db[uid].table(fill=True)
        for i in range(2, t.picam_image.size + 1):
            imgs.append(t.picam_image[i].squeeze())
    return np.array(imgs)

def get_image(uid):
    imgs = _get_images(uid)
    return np.mean(imgs, axis=0)

def get_spectrum(uid, nx1=410, nx2=460, ny1=1100, ny2=2000):
    image = get_image(uid)
    # spectrum = np.mean(image[510:560, 535:1000], axis=0)
    # nx1=510, nx2=560, ny1=535, ny2=1000
    spectrum = np.mean(image[nx1:nx2, ny1:ny2], axis=0)
    return spectrum

from scipy.signal import medfilt2d, medfilt

def get_spectrum_std(uid):
    images = _get_images(uid)
    spectra = np.mean(images[:, 510:560, 535:1000], axis=1)
    spectrum_std = np.std(spectra, axis=0)
    return spectrum_std

def get_diff_spectrum(uid_xray, uid_dark, **kwargs):
    spectrum_xray = get_spectrum(uid_xray, **kwargs)
    spectrum_dark = get_spectrum(uid_dark, **kwargs)
    return spectrum_xray - spectrum_dark

# xes_CaO = get_diff_spectrum('95e43edc-9507-44f1-966f-db923b7338f4', 'fc298c99-fd12-4ff6-b915-294e9efadf91')

# energy is 8000
xes_S = get_diff_spectrum('8e5dba0f-ac99-4ed5-bf1b-002638f807b0', 'fa05d9f4-8f17-4203-8074-8fba416829c0')
xes_NiS = get_diff_spectrum('ce9e1afe-e4d2-4d44-9aca-ee36137921c7', '001c7a48-a31d-4d31-ba10-9b087c9b0c20')
xes_CuSO4 = get_diff_spectrum('3091b648-593e-445c-b52f-928b26653493', 'c0919773-7097-42b8-a928-46a8cf4a8ca6')


# hopefully better statistics
xes_S = get_diff_spectrum('ed6d26af-4021-4948-bf34-37e9f83c67e4', 'dad47f8b-fb76-4750-8ed7-22b037562930')
xes_NiS = get_diff_spectrum('fdb767cc-352d-48df-bf6c-d481df424c0b', '1a3fe82d-9b93-4af0-b38a-fef11b94444a')

# energy is 6900

# xes_S = get_diff_spectrum('2b693ada-98da-43c0-84f7-94fe60f8f821', 'fa05d9f4-8f17-4203-8074-8fba416829c0')
# xes_CuSO4 = get_diff_spectrum('3bf4ae34-21b1-49d3-97eb-d18a1039c9cf', 'c0919773-7097-42b8-a928-46a8cf4a8ca6')


# plt.figure(); plt.imshow(get_image('3bf4ae34-21b1-49d3-97eb-d18a1039c9cf'))


def norm_y(y, n1=0, n2=10):
    # return (y - np.mean(y[80:100])) / (np.percentile(y, 98) - np.mean(y[:10]))
    return (y - np.mean(y[n1:n2])) / (np.max(y) - np.mean(y[n1:n2]))

plt.figure(2, clear=True)
plt.plot(norm_y(xes_S), label='S$^0$')
plt.plot(norm_y(xes_NiS), label='S$^{2-}$')
plt.plot(norm_y(xes_CuSO4), label='S$^{6+}$')
plt.legend()

# plt.plot(spectrum_dark)

data = [{'label': 'Low 4 MHz', 'spectrum': get_diff_spectrum(       'ed6d26af-4021-4948-bf34-37e9f83c67e4', 'dad47f8b-fb76-4750-8ed7-22b037562930')},
        {'label': 'Low 1 MHz', 'spectrum': get_diff_spectrum(       '3348b8eb-cc6e-497a-9b4e-3956baf9d930', '3db3d8d8-e6ff-4637-9cf2-61798c92a604')},
        {'label': 'Medium 4 MHz', 'spectrum': get_diff_spectrum(    '2bc9f5d0-9b7f-4028-adf7-ed4dc524e0d6', 'da240c40-1dc5-406d-ae3b-034923161309' )},
        {'label': 'Medium 1 MHz', 'spectrum': get_diff_spectrum(    '59b4c491-17c8-486c-b720-960c83a8369f', '76f75e5c-4fc8-41be-a8e1-7b2034298223')},
        {'label': 'High 4 MHz', 'spectrum': get_diff_spectrum(      'a0dc124c-213d-4400-8b2c-77be35067cbc','e6abc2c2-1099-4b5c-974f-87227d7c48b0')},
        {'label': 'High 1 MHz', 'spectrum': get_diff_spectrum(      '35846bb3-122d-487b-8c50-82f5e4bd7662', '23c569b4-a79c-4847-aba2-97d9c3baf421')}]

plt.figure(1, clear=True)
for d in data:
    plt.plot(norm_y(d['spectrum']), label=d['label'])
plt.legend()


uids_xray = ['ed6d26af-4021-4948-bf34-37e9f83c67e4',
'3348b8eb-cc6e-497a-9b4e-3956baf9d930',
'2bc9f5d0-9b7f-4028-adf7-ed4dc524e0d6',
'59b4c491-17c8-486c-b720-960c83a8369f',
'a0dc124c-213d-4400-8b2c-77be35067cbc',
'35846bb3-122d-487b-8c50-82f5e4bd7662',]

plt.figure(2, clear=True)
for uid, d in zip(uids_xray, data):
    _s = get_spectrum(uid)
    _s_std = get_spectrum_std(uid)

    # spectrum_std = get_spectrum_std(uid)
    plt.plot(_s_std/_s,label = d['label'])
plt.legend()



xes_CuSO4_200ms_x50 = get_diff_spectrum('26dec81f-d05e-41d0-b6cb-2399d13f8061', '5b0153b5-4800-4b82-8c52-db9e46932641')
xes_CuSO4_1s_x10 = get_diff_spectrum('bc9ab44d-f7ff-482e-add7-3ea9aae30f50', '4fec3cc0-852d-40d1-a129-a1fcb9df1422')
xes_CuSO4_10s_x1 = get_diff_spectrum('ba97a231-b4b9-4543-9fe8-3c287ded1e52', 'ac14d1c3-04b7-4068-b165-fce49795c269')

plt.figure(1, clear=True)
plt.plot(norm_y(xes_CuSO4_200ms_x50), label='200ms x50')
plt.plot(norm_y(xes_CuSO4_1s_x10)-0.5, label='1s x10')
plt.plot(norm_y(xes_CuSO4_10s_x1)-1.0, label='10s x1')
plt.legend()


plt.figure(2, clear=True)
plt.semilogx([0.2, 1, 10], [np.std(norm_y(xes_CuSO4_200ms_x50)[80:180]),
          np.std(norm_y(xes_CuSO4_1s_x10)[80:180]),
          np.std(norm_y(xes_CuSO4_10s_x1)[80:180])], 'ks-')
plt.xticks([0.2, 1, 10], ['0.2', '1', '10'])
plt.xlabel('exposure time')
plt.ylabel('readout noise')


xes_CuSO4_10s = get_diff_spectrum('02673265-b344-4895-bcee-b5aa4e204b62', 'ac14d1c3-04b7-4068-b165-fce49795c269', nx1=510, nx2=560, ny1=535, ny2=1000)
# xes_NiS_10s = get_diff_spectrum('8737790c-b258-46ee-8717-08d7117cdbd6', 'ac14d1c3-04b7-4068-b165-fce49795c269')
xes_NiS_10s = get_diff_spectrum('8d212734-0e0d-4974-97a6-a24d34b23daa', 'ac14d1c3-04b7-4068-b165-fce49795c269', nx1=510, nx2=560, ny1=535, ny2=1000)
xes_S_10s = get_diff_spectrum('a345643c-2ac3-48b2-9d68-f53153769736', 'ac14d1c3-04b7-4068-b165-fce49795c269', nx1=510, nx2=560, ny1=535, ny2=1000)


from scipy.interpolate import interp1d
energy_interp = interp1d([28.5, 265.5], [6900/3, 2308], fill_value='extrapolate')

plt.figure(1, clear=True)
plt.plot(energy_interp(np.arange(xes_CuSO4_10s.size)), norm_y(xes_CuSO4_10s, n1=60, n2=80), label='S$^{6+}$ (CuSO4)')
plt.plot(energy_interp(np.arange(xes_S_10s.size) + 2.5), norm_y(xes_S_10s, n1=60, n2=80), label='S$^{0}$ (S)')
plt.plot(energy_interp(np.arange(xes_NiS_10s.size) + 1), norm_y(xes_NiS_10s, n1=60, n2=80), label='S$^{2-}$ (NiS)')
# plt.plot(norm_y(xes_CuSO4_1s_x10)-0.5, label='1s x10')
# plt.plot(norm_y(xes_CuSO4_10s_x1)-1.0, label='10s x1')
plt.legend()

plt.xlim(2302, 2314)

plt.figure(2, clear=True)
plt.plot(np.arange(70), norm_y(xes_CuSO4_10s)[:70] / norm_y(xes_CuSO4_10s)[:70].max(), label='CuSO4')
plt.plot(np.arange(70)+2.5, norm_y(xes_S_10s)[:70] / norm_y(xes_S_10s)[:70].max(), label='S')
plt.plot(np.arange(70)+1, norm_y(xes_NiS_10s)[:70] / norm_y(xes_NiS_10s)[:70].max(), label='NiS')
# plt.plot(norm_y(xes_CuSO4_1s_x10)-0.5, label='1s x10')
# plt.plot(norm_y(xes_CuSO4_10s_x1)-1.0, label='10s x1')
plt.legend()

# visualizing S images
VMIN=20000
VMAX=30000
plt.figure(2, clear=True)
plt.subplot(131)
plt.imshow(get_image('a345643c-2ac3-48b2-9d68-f53153769736')[500:560, 685:900], vmin=VMIN, vmax=VMAX)
plt.title('X-rays ON')
plt.colorbar()

plt.subplot(132)
plt.imshow(get_image('ac14d1c3-04b7-4068-b165-fce49795c269')[500:560, 685:900], vmin=VMIN, vmax=VMAX)
plt.title('Dark')
plt.colorbar()

plt.subplot(133)
plt.imshow(get_image('a345643c-2ac3-48b2-9d68-f53153769736')[500:560, 685:900] -
           get_image('ac14d1c3-04b7-4068-b165-fce49795c269')[500:560, 685:900], vmin=0, vmax=8000)
plt.title('Difference')
plt.colorbar()
plt.tight_layout()
xray_image = picam.image.array_data.get().reshape(1024, 2048)
xray_image2 = picam.image.array_data.get().reshape(1024, 2048)
dark_image = picam.image.array_data.get().reshape(1024, 2048)

plt.figure(); plt.imshow(xray_image + xray_image2 - 2*dark_image, vmin=0, vmax=2000)


xes_Ag = get_diff_spectrum('97d5ec49-1012-4181-9d70-ff7bc5914bba', '6be5c7da-1bb1-406b-a198-c4b35ffcd939')

# xes_PdO = get_diff_spectrum('f5abe7c0-8665-4b56-88d6-fd3b4ae0f8c7', '88c05f17-1f3f-483e-9d98-ff2877997b08')
xes_PdO = get_diff_spectrum('4ce40b1b-34af-42ce-a8c1-09590c248ebe', 'a9e5b448-bcbc-4bf6-970e-e7c879dce3da')
xes_Pd = get_diff_spectrum('602a4a09-c573-4ca5-a6df-2179c247b99c', '561198a8-c738-494f-b298-84cc6fe7c8e5')
#Pd elastic scans: 6670 - '10587276-105a-413f-b021-64a58a2639f6', 6640 - '056c3a42-b10c-48bf-a63f-7c71dc86852c'

xes_PdCl2 = get_diff_spectrum('566a26db-7d27-4d9c-8e46-5cab8c788bb1', 'a35794f5-25e2-46b7-a137-462fe7ac0448')
xes_12Z23IE = get_diff_spectrum('4a480d33-efb0-4538-9dcd-c9024d0c0eee', '127ea31d-c1fd-4a95-889b-511c75990576')


plt.figure(2, clear=True)
# plt.plot(xes_Ag)

plt.plot(norm_y(xes_Pd), label='Pd')
plt.plot(norm_y(xes_PdO), label='PdO')
# plt.plot(norm_y(xes_PdCl2), label='PdCl2')
plt.plot(norm_y(xes_12Z23IE), label='Z23IE')
plt.legend()



daq_dict = [{'sample_name': 'PdCl2', 'position':{'x': -12.520,
                                                'y': -42.992,
                                                'z': -12.604}},
            {'sample_name': 'Pd(NH3)4', 'position': {'x': -12.220,
                                                     'y': -58.532,
                                                     'z': -9.504}},

            {'sample_name': 'Pd2', 'position': {'x': -12.220,
                                                'y': -71.342,
                                                'z': -9.804}},

            {'sample_name': 'Pd3', 'position': {'x': -12.220,
                                                'y': -86.592,
                                                'z': -11.304}},

            ]

while True:
    for sample in daq_dict:
        RE(move_sample_stage_plan(sample['position']))
        md = {**sample}



        md['image'] = 'dark'
        RE(bp.count([picam], 6, md=md))

        shutter.open()

        md['image'] = 'data'
        RE(move_mono_energy(7000))
        RE(bp.count([picam], 11, md=md))

        md['image'] = '6640'
        RE(move_mono_energy(6640))
        RE(bp.count([picam], 2, md=md))

        md['image'] = '6670'
        RE(move_mono_energy(6670))
        RE(bp.count([picam], 2, md=md))

        shutter.close()



# visualizing  images
VMIN=10000
VMAX=20000

# Pd
# dark_uids = '602a4a09-c573-4ca5-a6df-2179c247b99c'
# bright_uids = '561198a8-c738-494f-b298-84cc6fe7c8e5'

# PdO
# dark_uids = 'a9e5b448-bcbc-4bf6-970e-e7c879dce3da'
# bright_uids = '4ce40b1b-34af-42ce-a8c1-09590c248ebe'

# Pd_6640
# bright_uids = '056c3a42-b10c-48bf-a63f-7c71dc86852c'
# dark_uids = 'a9e5b448-bcbc-4bf6-970e-e7c879dce3da'

# PdCl2
# bright_uids = '566a26db-7d27-4d9c-8e46-5cab8c788bb1'
# dark_uids = 'a35794f5-25e2-46b7-a137-462fe7ac0448'

# bright_uids = ['566a26db-7d27-4d9c-8e46-5cab8c788bb1',
#                '7347de44-6f96-4951-9e02-9f8356f1cca9',
#                '8fa81c46-b454-48a4-9053-ce4c0aab978c',
#                '054b9406-5236-407e-bee5-3cef1736c06f',
#                'b8a09724-2bb3-49d2-a33b-33c4287d3f2e',
#                '50b91bbe-fd17-4c93-a3e9-6a655a733520',
#                '45341693-e724-4693-80b7-23f122857bcc',
#                '2648a34d-416c-495d-9f92-b86eba976a69',
#                '77a1d220-c711-4eb5-bba2-e1037d44d7bd']
#
#
# dark_uids = ['a35794f5-25e2-46b7-a137-462fe7ac0448',
#              '3fad56fd-b17f-4733-b08a-28333c33b4ae',
#              'b9aca12a-309c-4cba-882b-62080026d483',
#              '55c02ae4-af9e-4997-8722-f1dec2cbb2f7',
#              'bb1505e6-d740-4d40-b50e-91c1347018ae',
#              'ac87f9e2-1fbf-4a2f-bf49-493266d2f5e2',
#              '90c7e76a-835c-42d1-bf86-26230b8f618a',
#              'bbcad194-f9ae-445c-8afd-a16125b96b81',
#              '22b18e71-b6e4-4a27-a00f-d4bbbcf6878f',
#              '89ae0563-0833-4bfb-a5a0-952cd52b42b8']

# PdCl2 elastic
# bright_uids = ['1992dd7b-7986-42cd-a563-6b8e9f319e9a',
#  '67ae327a-81fc-4e0a-8d50-3e17736b6fec',
#  '5ba41985-4ae8-4553-95bb-c28119edb8c7',
#  '4d7422f9-f982-46df-ad1d-8c73cbc81cbf',
#  '35b78814-f7a4-4c7b-ba0e-2dc3a2001cd2',
#  '06d0826d-a0ad-4b87-b397-1701230979f8',
#  'aaea69f8-0e22-47bb-b43e-8d65d9655fd3',
#  '08fdf135-7cb8-4dfb-b88f-7e621b4820f2',
#  'ab20c160-2d36-42d9-8353-7fe9e3ec0721']
#
#
# dark_uids = ['4832393a-4c90-4216-a336-afe2baaae22f',
#  '851579b3-79db-4927-a770-d8c472ae813d',
#  'e2301d15-0466-4beb-b1ab-b531ef131288',
#  'f3fcf316-fb5d-4158-8cfd-9448afbaed99',
#  'b4dcde49-d414-4bee-9ddd-e1441238d52f',
#  'f81cdac0-af65-4a91-b3aa-2871ff8880ea',
#  'a53c7d82-4421-46cf-bafb-ccd60de17771',
#  '17110bfe-b73a-48d0-9ca0-0a998f7c717d',
#  '71d650b4-2ac8-48c1-9361-f433826102f8']

# Pd(NH3)4
# bright_uids = ['c3dfdff3-bf39-4732-a973-522a24f4e6a2',
#  'e5f0157b-0239-41a0-8197-15e83d0c7b75',
#  'd0651960-1cd9-4dd7-b8a2-81c058c1085f',
#  '39470b7c-c7c7-42d4-8d92-b8be21517323',
#  'c3f8e77d-5b0d-41fb-aac4-0e1a97529878',
#  '47f756cd-413c-4d8a-9e6c-3fed1e8b7c83',
#  '08d0def3-2239-4ad5-946d-c9a2281ac474',
#  '3a874c4b-b724-44aa-94ac-806bdff9a024',
#  '155e8363-8b42-4b1d-ba14-264a043fc61f']
#
# dark_uids=['9173c023-843d-4fdf-b36a-46b7ba65a53b',
#  'e96e70be-eb75-4711-a655-faf67fca8890',
#  '0677f1e9-05f3-4ddb-a330-de0264701438',
#  '326a8be0-82c3-4ea2-ac9d-96bba75df16f',
#  'ca7be1b8-e532-4273-8d17-05921efafc8b',
#  '78082826-de89-4692-8504-ee28c66117b2',
#  'dd31c57c-1e55-4cd4-999c-90227dc53826',
#  'e004037d-e7ec-4a9d-884a-6ba7536aff72',
#  '1e9292b6-9005-42f7-b3f2-a9d49ee05e72']

# # Pd(NH3)4 elastic
# bright_uids = ['35ab3f72-4091-4560-83e2-5b4b2f6effb1',
#  '9b639407-be56-4c43-8dba-fdaa954745c4',
#  '928a95ca-6555-41dc-b203-d4d75d2bfa3d',
#  '6bf5a9cd-2e8c-401d-92a3-d3bed6172a8b',
#  '604b68c2-3230-46ad-a91a-80662240e818',
#  '7ca5d3ad-d151-40f8-abeb-8e3318a4ac8e',
#  '69ac54c3-d2f9-42a1-95ff-df988e34b1d1',
#  '3afca479-7be2-40bf-abad-e760edd64dfe',
#  '6ca9689e-e403-4c26-bfb9-fe918a0da0e1']
#
# dark_uids=['4f9862e5-7814-43d7-a7a1-3367e04c7901',
#  'ab6fb0b4-0e24-4d28-aa1f-aef9024ed19c',
#  '20319d5e-63d8-486d-a2fb-88b1efc21ac1',
#  '514cba99-4588-457e-a829-bbf9a11454f5',
#  'a5b9e34b-0b15-4cc2-8bcb-3f0a9f81a4a5',
#  'aefe2c4d-7215-45c4-bf86-47032b15a502',
#  'd1477acb-a64f-4f58-8b01-0bde7790271e',
#  'adb891d5-cff6-4462-b8e2-5e9b933bc030',
#  '3e92c745-8b99-47ea-b3af-cc89661e8a23']

# Pd2
# bright_uids =['e14aa64c-27f5-4ff1-b1fd-5f7ba50ec7e7',
#  'db4fb6f6-1b3e-4bd7-9607-2d26b8597adc',
#  '6f2ecf29-aa8c-4a39-a646-161e4ee8ed67',
#  '4b8ef676-7ed2-4ec1-8f86-2ae335b6d8c2',
#  '8d675274-b0e0-4eb5-98b3-abf653328c6c',
#  '526aaab2-5e0f-4753-ad0e-846bdd3bdfcd',
#  '5d9f02cf-6981-4ead-828b-77556204c58e',
#  '31a40963-83a7-4999-bc0a-6ee1672d3e83',
#  'dfc54abc-d3e9-440f-bb73-80223dcf499a']
#
#
# dark_uids =['d65afa97-4908-4bd3-a898-b0b500ea2da9',
#  'e8c5cc0d-e648-4413-b003-590ab5f7bd66',
#  'a45e9b96-c746-4702-b832-8bc700cca72f',
#  '617ede9d-cc22-4862-9f94-6430de1ab957',
#  '8c5deba6-7ab7-490d-87b1-96621310da21',
#  'ac0c1bd8-a7fc-4e4e-92f6-c05d11e37b61',
#  '9fd5e67a-c493-400d-aef4-ac90a21c3e0f',
#  '75b5ebf8-37e7-412c-a61b-817a9dd56882',
#  'ca84c094-cf61-449c-9a03-09c08de242df']


# # Pd2 elastic
# bright_uids =['57ac501c-fbc9-489a-9fe4-cf985b5ce88d',
#  '9dd11765-4e27-4c1e-bb36-692a278b5b94',
#  'c54e71b7-9457-4933-9d16-9c6770ecd5b0',
#  '5bd922e2-3027-4a30-8d90-64a9075707c0',
#  '31df1441-6ae2-4e10-847e-5f3a356163ca',
#  'd1b88cb4-355c-43db-9520-7b64c2072cd2',
#  '9ea26ba2-84b3-4227-af3a-b9745f41c80d',
#  '30833947-cabb-4f8d-af2b-5e359fb19933',
#  '48a69ad7-76b6-4061-a6f3-d0668a01477a']
#
# dark_uids =['f48972b5-f829-4342-a8ba-8a1a1ac53ed5',
#  'c5131f36-eb40-4738-bd52-366ff94e8135',
#  'cec03d1a-5649-41bc-86ae-2a58f56b86fd',
#  '4ee97ba0-cc40-4170-a5ce-b2549df0b6ea',
#  'ea645434-b7d2-44c4-b6f2-372332e126ba',
#  '24a00ecf-2722-42df-b687-9ff51f1fb78c',
#  'b8204e28-040a-4139-aa63-839327fd0d00',
#  '3b675eb1-2571-45b6-a8d8-cb5d0fa27df9',
#  'd14f0abd-f016-4b15-a376-18f20b502166']


# Pd3
bright_uids =['6b128517-bfe4-40d0-94cf-def3ae979ea1',
 '34906393-827a-43f1-9863-9ee8525d09ee',
 '916917fa-6277-466d-b8ac-14f490ec5b30',
 '94128351-f342-41ef-a93d-081f6d5c670f',
 'd038ee63-e076-4d07-a870-6876d511d124',
 'dac0c16b-32e9-487c-964a-9d089227e6ca',
 '578bc61c-7c1d-4966-8b70-803360cee9fc',
 'fdc035a2-a090-47f1-8afd-f022ca8d1c34',
 '34a7b9be-0717-45ba-84ea-f02baa3d7294']

dark_uids =['187f522c-f9ff-42c1-bd8a-21f2d03f1af2',
 'd4d1e822-6c7b-4b6f-874b-d10bcb8888f7',
 '235eefc2-2011-48e3-90fa-260e49ca7445',
 '524e87d2-4b83-41c8-b8e3-02cc65768275',
 '578ee294-79df-4772-b981-d3df2b6be59d',
 'ef7462ed-4ac3-470f-b7d4-63188e846bc7',
 '3352a16a-534e-4821-89c1-74a5441bffcb',
 '3e73d638-f16b-430e-8875-84e8dd201795',
 '7b2d99b6-d35c-4ab5-a3d9-063835407664']



# # # Pd3 elastic
bright_uids =['127ea31d-c1fd-4a95-889b-511c75990576',
 '4a480d33-efb0-4538-9dcd-c9024d0c0eee',
 'd28dd5f7-72d6-4f2b-b28d-a412ee1b8a56',
 '07479908-d619-4bbf-a1ff-53d73e938f5b',
 'a5d210fe-fa1d-4be2-ade4-7dfdbc21c288',
 '5ae0fca2-8eda-42ef-9f13-67c92884d50b',
 '4eb1c088-5109-4abb-acd7-2a706ea21e5d',
 '94a322fc-db1a-40e6-a35d-a62b372e1348',
 '7f0637d9-449a-4406-a331-d442fa5586b9',
 'e068e24f-af07-48ac-acec-8c5988ba003f',
 'd0d902cd-0939-487e-8fae-856d72526812',
 '85f83507-a243-4507-918a-c54cbaa34e34',
 'cb6ddfc5-2ada-4310-ac3b-5b2bb559807a',
 'c6a6d288-7a62-41a1-a1b3-dd16328210c2']


dark_uids =['578d23ff-f35d-45af-8fe9-1f7f18bdefb9',
 '82604fd3-bd12-4366-aeb4-dae245bb6b55',
 '2e4e71c1-7ee9-47f0-858e-e95e8a513f4d',
 'f11a57de-b661-46a4-8c6b-3477704c55f4',
 '93c16531-0a2b-458c-9dc5-38d6445b0cc3',
 '9b37d3fb-bdad-4eba-acbe-d87f0606dada',
 '435fa16d-5378-441a-8041-7a0f0419d772',
 'd91d47f4-f222-40be-ad92-39fd218d48b9']


image_bright = get_image(bright_uids)
image_dark = get_image(dark_uids)
nx1, nx2 = 1, 1024
ny1, ny2 = 1, 2048

# nx1, nx2 = 380, 480
# ny1, ny2 = 600, 2048


plt.figure(2, clear=True)
plt.subplot(311)

plt.imshow(image_bright[nx1:nx2, ny1:ny2], vmin=VMIN, vmax=VMAX)
plt.title('X-rays ON')
# plt.colorbar()

plt.subplot(312)
plt.imshow(image_dark[nx1:nx2, ny1:ny2], vmin=VMIN, vmax=VMAX)
# plt.imshow(medfilt2d(get_image(dark_uids)[nx1:nx2, ny1:ny2]), vmin=VMIN, vmax=VMAX)
plt.title('Dark')
# plt.colorbar()

plt.subplot(313)
plt.imshow(image_bright[nx1:nx2, ny1:ny2] -
           image_dark[nx1:nx2, ny1:ny2], vmin=-2000, vmax=2000)
plt.title('Difference')
# plt.colorbar()
# plt.tight_layout()



xes_PdO = get_diff_spectrum('4ce40b1b-34af-42ce-a8c1-09590c248ebe', 'a9e5b448-bcbc-4bf6-970e-e7c879dce3da', nx1=380, nx2=480, ny1=500, ny2=2048)
xes_Pd = get_diff_spectrum('602a4a09-c573-4ca5-a6df-2179c247b99c', '561198a8-c738-494f-b298-84cc6fe7c8e5', nx1=380, nx2=480, ny1=500, ny2=2048)
# xes_PdCl2 = get_diff_spectrum('566a26db-7d27-4d9c-8e46-5cab8c788bb1', 'a35794f5-25e2-46b7-a137-462fe7ac0448', nx1=380, nx2=480, ny1=500, ny2=2048)
xes_PdCl2 = get_diff_spectrum(['566a26db-7d27-4d9c-8e46-5cab8c788bb1',
                               '7347de44-6f96-4951-9e02-9f8356f1cca9',
                               '8fa81c46-b454-48a4-9053-ce4c0aab978c',
                               '054b9406-5236-407e-bee5-3cef1736c06f',
                               'b8a09724-2bb3-49d2-a33b-33c4287d3f2e',
                               '50b91bbe-fd17-4c93-a3e9-6a655a733520',
                               '45341693-e724-4693-80b7-23f122857bcc',
                               '2648a34d-416c-495d-9f92-b86eba976a69',
                               '77a1d220-c711-4eb5-bba2-e1037d44d7bd'],
                              ['a35794f5-25e2-46b7-a137-462fe7ac0448',
                               '3fad56fd-b17f-4733-b08a-28333c33b4ae',
                               'b9aca12a-309c-4cba-882b-62080026d483',
                               '55c02ae4-af9e-4997-8722-f1dec2cbb2f7',
                               'bb1505e6-d740-4d40-b50e-91c1347018ae',
                               'ac87f9e2-1fbf-4a2f-bf49-493266d2f5e2',
                               '90c7e76a-835c-42d1-bf86-26230b8f618a',
                               'bbcad194-f9ae-445c-8afd-a16125b96b81',
                               '22b18e71-b6e4-4a27-a00f-d4bbbcf6878f',
                               '89ae0563-0833-4bfb-a5a0-952cd52b42b8'],
                              nx1=380, nx2=480, ny1=500, ny2=2048)
xes_PdNH34 = get_diff_spectrum(['c3dfdff3-bf39-4732-a973-522a24f4e6a2',
                                'e5f0157b-0239-41a0-8197-15e83d0c7b75',
                                'd0651960-1cd9-4dd7-b8a2-81c058c1085f',
                                '39470b7c-c7c7-42d4-8d92-b8be21517323',
                                'c3f8e77d-5b0d-41fb-aac4-0e1a97529878',
                                '47f756cd-413c-4d8a-9e6c-3fed1e8b7c83',
                                '08d0def3-2239-4ad5-946d-c9a2281ac474',
                                '3a874c4b-b724-44aa-94ac-806bdff9a024',
                                '155e8363-8b42-4b1d-ba14-264a043fc61f'],
                               ['9173c023-843d-4fdf-b36a-46b7ba65a53b',
                                'e96e70be-eb75-4711-a655-faf67fca8890',
                                '0677f1e9-05f3-4ddb-a330-de0264701438',
                                '326a8be0-82c3-4ea2-ac9d-96bba75df16f',
                                'ca7be1b8-e532-4273-8d17-05921efafc8b',
                                '78082826-de89-4692-8504-ee28c66117b2',
                                'dd31c57c-1e55-4cd4-999c-90227dc53826',
                                'e004037d-e7ec-4a9d-884a-6ba7536aff72',
                                '1e9292b6-9005-42f7-b3f2-a9d49ee05e72'],
                              nx1=380, nx2=480, ny1=500, ny2=2048)

xes_Pd2 = get_diff_spectrum(['e14aa64c-27f5-4ff1-b1fd-5f7ba50ec7e7',
 'db4fb6f6-1b3e-4bd7-9607-2d26b8597adc',
 '6f2ecf29-aa8c-4a39-a646-161e4ee8ed67',
 '4b8ef676-7ed2-4ec1-8f86-2ae335b6d8c2',
 '8d675274-b0e0-4eb5-98b3-abf653328c6c',
 '526aaab2-5e0f-4753-ad0e-846bdd3bdfcd',
 '5d9f02cf-6981-4ead-828b-77556204c58e',
 '31a40963-83a7-4999-bc0a-6ee1672d3e83',
 'dfc54abc-d3e9-440f-bb73-80223dcf499a'],
                            ['d65afa97-4908-4bd3-a898-b0b500ea2da9',
                             'e8c5cc0d-e648-4413-b003-590ab5f7bd66',
                             'a45e9b96-c746-4702-b832-8bc700cca72f',
                             '617ede9d-cc22-4862-9f94-6430de1ab957',
                             '8c5deba6-7ab7-490d-87b1-96621310da21',
                             'ac0c1bd8-a7fc-4e4e-92f6-c05d11e37b61',
                             '9fd5e67a-c493-400d-aef4-ac90a21c3e0f',
                             '75b5ebf8-37e7-412c-a61b-817a9dd56882',
                             'ca84c094-cf61-449c-9a03-09c08de242df'],
                              nx1=380, nx2=480, ny1=500, ny2=2048)

xes_Pd3 = get_diff_spectrum(['6b128517-bfe4-40d0-94cf-def3ae979ea1',
 '34906393-827a-43f1-9863-9ee8525d09ee',
 '916917fa-6277-466d-b8ac-14f490ec5b30',
 '94128351-f342-41ef-a93d-081f6d5c670f',
 'd038ee63-e076-4d07-a870-6876d511d124',
 'dac0c16b-32e9-487c-964a-9d089227e6ca',
 '578bc61c-7c1d-4966-8b70-803360cee9fc',
 'fdc035a2-a090-47f1-8afd-f022ca8d1c34',
 '34a7b9be-0717-45ba-84ea-f02baa3d7294'],
                            ['187f522c-f9ff-42c1-bd8a-21f2d03f1af2',
                             'd4d1e822-6c7b-4b6f-874b-d10bcb8888f7',
                             '235eefc2-2011-48e3-90fa-260e49ca7445',
                             '524e87d2-4b83-41c8-b8e3-02cc65768275',
                             '578ee294-79df-4772-b981-d3df2b6be59d',
                             'ef7462ed-4ac3-470f-b7d4-63188e846bc7',
                             '3352a16a-534e-4821-89c1-74a5441bffcb',
                             '3e73d638-f16b-430e-8875-84e8dd201795',
                             '7b2d99b6-d35c-4ab5-a3d9-063835407664'],
                              nx1=380, nx2=480, ny1=500, ny2=2048)




#Pd elastic scans: 6670 - '10587276-105a-413f-b021-64a58a2639f6', 6640 - '056c3a42-b10c-48bf-a63f-7c71dc86852c'
xes_Pd_elastic = get_diff_spectrum('056c3a42-b10c-48bf-a63f-7c71dc86852c', '10587276-105a-413f-b021-64a58a2639f6', nx1=380, nx2=480, ny1=500, ny2=2048)
# xes_Pd_6670 = get_diff_spectrum(, '561198a8-c738-494f-b298-84cc6fe7c8e5', nx1=380, nx2=480, ny1=500, ny2=2048)
xes_PdCl2_elastic = get_diff_spectrum(['1992dd7b-7986-42cd-a563-6b8e9f319e9a',
                                       '67ae327a-81fc-4e0a-8d50-3e17736b6fec',
                                       '5ba41985-4ae8-4553-95bb-c28119edb8c7',
                                       '4d7422f9-f982-46df-ad1d-8c73cbc81cbf',
                                       '35b78814-f7a4-4c7b-ba0e-2dc3a2001cd2',
                                       '06d0826d-a0ad-4b87-b397-1701230979f8',
                                       'aaea69f8-0e22-47bb-b43e-8d65d9655fd3',
                                       '08fdf135-7cb8-4dfb-b88f-7e621b4820f2',
                                       'ab20c160-2d36-42d9-8353-7fe9e3ec0721'],
                                      ['4832393a-4c90-4216-a336-afe2baaae22f',
                                       '851579b3-79db-4927-a770-d8c472ae813d',
                                       'e2301d15-0466-4beb-b1ab-b531ef131288',
                                       'f3fcf316-fb5d-4158-8cfd-9448afbaed99',
                                       'b4dcde49-d414-4bee-9ddd-e1441238d52f',
                                       'f81cdac0-af65-4a91-b3aa-2871ff8880ea',
                                       'a53c7d82-4421-46cf-bafb-ccd60de17771',
                                       '17110bfe-b73a-48d0-9ca0-0a998f7c717d',
                                       '71d650b4-2ac8-48c1-9361-f433826102f8'], nx1=380, nx2=480, ny1=500, ny2=2048)
xes_PdNH34_elastic = get_diff_spectrum(['35ab3f72-4091-4560-83e2-5b4b2f6effb1',
                                        '9b639407-be56-4c43-8dba-fdaa954745c4',
                                        '928a95ca-6555-41dc-b203-d4d75d2bfa3d',
                                        '6bf5a9cd-2e8c-401d-92a3-d3bed6172a8b',
                                        '604b68c2-3230-46ad-a91a-80662240e818',
                                        '7ca5d3ad-d151-40f8-abeb-8e3318a4ac8e',
                                        '69ac54c3-d2f9-42a1-95ff-df988e34b1d1',
                                        '3afca479-7be2-40bf-abad-e760edd64dfe',
                                        '6ca9689e-e403-4c26-bfb9-fe918a0da0e1'],
                                       ['4f9862e5-7814-43d7-a7a1-3367e04c7901',
                                        'ab6fb0b4-0e24-4d28-aa1f-aef9024ed19c',
                                        '20319d5e-63d8-486d-a2fb-88b1efc21ac1',
                                        '514cba99-4588-457e-a829-bbf9a11454f5',
                                        'a5b9e34b-0b15-4cc2-8bcb-3f0a9f81a4a5',
                                        'aefe2c4d-7215-45c4-bf86-47032b15a502',
                                        'd1477acb-a64f-4f58-8b01-0bde7790271e',
                                        'adb891d5-cff6-4462-b8e2-5e9b933bc030',
                                        '3e92c745-8b99-47ea-b3af-cc89661e8a23'], nx1=380, nx2=480, ny1=500, ny2=2048)

xes_Pd2_elastic = get_diff_spectrum(['57ac501c-fbc9-489a-9fe4-cf985b5ce88d',
 '9dd11765-4e27-4c1e-bb36-692a278b5b94',
 'c54e71b7-9457-4933-9d16-9c6770ecd5b0',
 '5bd922e2-3027-4a30-8d90-64a9075707c0',
 '31df1441-6ae2-4e10-847e-5f3a356163ca',
 'd1b88cb4-355c-43db-9520-7b64c2072cd2',
 '9ea26ba2-84b3-4227-af3a-b9745f41c80d',
 '30833947-cabb-4f8d-af2b-5e359fb19933',
 '48a69ad7-76b6-4061-a6f3-d0668a01477a'],
                                    ['f48972b5-f829-4342-a8ba-8a1a1ac53ed5',
                                     'c5131f36-eb40-4738-bd52-366ff94e8135',
                                     'cec03d1a-5649-41bc-86ae-2a58f56b86fd',
                                     '4ee97ba0-cc40-4170-a5ce-b2549df0b6ea',
                                     'ea645434-b7d2-44c4-b6f2-372332e126ba',
                                     '24a00ecf-2722-42df-b687-9ff51f1fb78c',
                                     'b8204e28-040a-4139-aa63-839327fd0d00',
                                     '3b675eb1-2571-45b6-a8d8-cb5d0fa27df9',
                                     'd14f0abd-f016-4b15-a376-18f20b502166'], nx1=380, nx2=480, ny1=500, ny2=2048)

xes_Pd3_elastic = get_diff_spectrum(['127ea31d-c1fd-4a95-889b-511c75990576',
 '4a480d33-efb0-4538-9dcd-c9024d0c0eee',
 'd28dd5f7-72d6-4f2b-b28d-a412ee1b8a56',
 '07479908-d619-4bbf-a1ff-53d73e938f5b',
 'a5d210fe-fa1d-4be2-ade4-7dfdbc21c288',
 '5ae0fca2-8eda-42ef-9f13-67c92884d50b',
 '4eb1c088-5109-4abb-acd7-2a706ea21e5d',
 '94a322fc-db1a-40e6-a35d-a62b372e1348',
 '7f0637d9-449a-4406-a331-d442fa5586b9',
 'e068e24f-af07-48ac-acec-8c5988ba003f',
 'd0d902cd-0939-487e-8fae-856d72526812',
 '85f83507-a243-4507-918a-c54cbaa34e34',
 'cb6ddfc5-2ada-4310-ac3b-5b2bb559807a',
 'c6a6d288-7a62-41a1-a1b3-dd16328210c2'],
                                    ['578d23ff-f35d-45af-8fe9-1f7f18bdefb9',
                                     '82604fd3-bd12-4366-aeb4-dae245bb6b55',
                                     '2e4e71c1-7ee9-47f0-858e-e95e8a513f4d',
                                     'f11a57de-b661-46a4-8c6b-3477704c55f4',
                                     '93c16531-0a2b-458c-9dc5-38d6445b0cc3',
                                     '9b37d3fb-bdad-4eba-acbe-d87f0606dada',
                                     '435fa16d-5378-441a-8041-7a0f0419d772',
                                     'd91d47f4-f222-40be-ad92-39fd218d48b9'], nx1=380, nx2=480, ny1=500, ny2=2048)


# xes_PdCl2 = get_diff_spectrum('566a26db-7d27-4d9c-8e46-5cab8c788bb1', 'a35794f5-25e2-46b7-a137-462fe7ac0448')
# xes_12Z23IE = get_diff_spectrum('4a480d33-efb0-4538-9dcd-c9024d0c0eee', '127ea31d-c1fd-4a95-889b-511c75990576')


def norm_bkg_y(_y, n1=10, n2=20, do_medfilt=True):
    if do_medfilt:
        y = medfilt(_y, 3)
    else:
        y = _y.copy()
    # return (y - np.mean(y[80:100])) / (np.percentile(y, 98) - np.mean(y[:10]))
    x = np.arange(y.size)
    x1, x2 = np.mean(x[:n1]), np.mean(x[-n2:])
    y1, y2 = np.mean(y[:n1]), np.mean(y[-n2:])
    p = np.polyfit([x1, x2], [y1, y2], 1)
    y_bkg = np.polyval(p, x)
    return (y - y_bkg) / (np.max(y) - y_bkg)

energy_interp = interp1d([703, 1338], [6640/2, 6670/2], fill_value='extrapolate')


# plt.plot(energy_interp(np.arange(xes_CuSO4_10s.size)), norm_y(xes_CuSO4_10s, n1=60, n2=80), label='S$^{6+}$ (CuSO4)')


plt.figure(2, clear=True)
# plt.plot(xes_Ag)

plt.plot(energy_interp(np.arange(xes_Pd.size)), norm_bkg_y(xes_Pd), label='Pd')
plt.plot(energy_interp(np.arange(xes_PdO.size)), norm_bkg_y(xes_PdO), label='PdO')
# plt.plot(energy_interp(np.arange(xes_PdCl2.size)+11), norm_bkg_y(xes_PdCl2), label='PdCl2')
plt.plot(energy_interp(np.arange(xes_PdNH34.size)+6), norm_bkg_y(xes_PdNH34), label='Pd(NH3)4')
# plt.plot(energy_interp(np.arange(xes_Pd2.size)+5), norm_bkg_y(xes_Pd2), label='Pd2')
# plt.plot(energy_interp(np.arange(xes_Pd3.size)+1), norm_bkg_y(xes_Pd3), label='Pd3')

# plt.plot(energy_interp(np.arange(xes_Pd_elastic.size)), norm_bkg_y(np.abs(xes_Pd_elastic)), label='Pd elastic')
# plt.plot(energy_interp(np.arange(xes_PdCl2_elastic.size)+11), norm_bkg_y(np.abs(xes_PdCl2_elastic)), label='PdCL2 elastic')
# plt.plot(energy_interp(np.arange(xes_PdNH34_elastic.size)+6), norm_bkg_y(np.abs(xes_PdNH34_elastic)), label='Pd(NH3)4 elastic')
# plt.plot(energy_interp(np.arange(xes_Pd2_elastic.size)+5), norm_bkg_y(np.abs(xes_Pd2_elastic)), label='Pd2 elastic')
# plt.plot(energy_interp(np.arange(xes_Pd3_elastic.size)+1), norm_bkg_y(np.abs(xes_Pd3_elastic)), label='Pd3 elastic')
# plt.plot(xes_Pd_6670, label='6670')
plt.xlim(3320, 3335)

plt.legend()
plt.xlabel('Emission energy, eV')
plt.ylabel('Intensity, a.u.')
plt.title('incident energy 7000 eV')





##################

from ophyd import EpicsMotor as _EpicsMotor
from ophyd import (Device, Kind, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, Kind,
                   PseudoPositioner, PseudoSingle, SoftPositioner, Signal, SignalRO)
from ophyd.status import SubscriptionStatus, DeviceStatus

class EpicsMotorWithTweaking(_EpicsMotor):
    # set does not work in this class; use put!
    twv = Cpt(EpicsSignal, '.TWV', kind='omitted')
    twr = Cpt(EpicsSignal, '.TWR', kind='omitted')
    twf = Cpt(EpicsSignal, '.TWF', kind='omitted')

EpicsMotor = EpicsMotorWithTweaking

##
import threading
motor_cr_main_roll = EpicsMotor('XF:08IDB-OP{HRS:1-Stk:1:Roll}Mtr', name='motor_cr_main_roll')
motor_cr_aux2_roll = EpicsMotor('XF:08IDB-OP{HRS:1-Stk:2:Roll}Mtr', name='motor_cr_aux2_roll')
motor_cr_aux3_roll = EpicsMotor('XF:08IDB-OP{HRS:1-Stk:3:Roll}Mtr', name='motor_cr_aux3_roll')

apb_timestamp_s = EpicsSignal('XF:08IDB-CT{PBA:1}:EVR:TS:Sec-I', name='apb_timestamp_s')
apb_timestamp_ns = EpicsSignal('XF:08IDB-CT{PBA:1}:EVR:TS:NSec-I', name='apb_timestamp_ns')

class FlyableEpicsMotor(Device):

    def __init__(self, motor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = motor
        self.traj_dict = None
        self.flying_status = None

    def set_trajectory(self, traj_dict):
        # traj_dict = {'positions': [point1, point2, point3, point4],
        #              'velocities': [v1_2, v2_3 ,v3_4]}
        self.traj_dict = traj_dict

    def prepare(self):
        return self.motor.move(self.traj_dict['positions'][0], wait=False)

    def kickoff(self):
        self.flying_status = DeviceStatus(self)
        thread = threading.Thread(target=self.execute_motion, daemon=True)
        thread.start()
        return self.flying_status

    def execute_motion(self):
        self.data = []
        def write_data_callback(value, timestamp, **kwargs):
            self.data.append([timestamp, value])
        cid = self.motor.user_readback.subscribe(write_data_callback)

        pre_fly_velocity = self.motor.velocity.get()
        for prev_position, next_position, duration in zip(self.traj_dict['positions'][:-1],
                                                          self.traj_dict['positions'][1:],
                                                          self.traj_dict['durations']):
            velocity = abs(next_position - prev_position) / duration
            self.motor.velocity.set(velocity).wait()
            self.motor.move(next_position).wait()
        self.flying_status.set_finished()

        self.motor.velocity.set(pre_fly_velocity).wait()
        self.motor.user_readback.unsubscribe(cid)


    def complete(self):
        self.flying_status = None
        self.traj_dict = None

    @property
    def current_trajectory_duration(self):
        return sum(self.traj_dict['durations']) + 5

flying_motor_cr_main_roll = FlyableEpicsMotor(johann_emission.motor_cr_main_roll, name='flying_motor_cr_main_roll')
flying_motor_cr_aux2_roll = FlyableEpicsMotor(johann_emission.motor_cr_aux2_roll, name='flying_motor_cr_aux2_roll')
flying_motor_cr_aux3_roll = FlyableEpicsMotor(johann_emission.motor_cr_aux3_roll, name='flying_motor_cr_aux3_roll')


pos_dict = johann_emission._forward({'energy' : 7085})

traj_dict_main = {'positions': [pos_dict['motor_cr_main_roll'] - 1000,
                                pos_dict['motor_cr_main_roll'] + 1000], 'durations': [30]}
flying_motor_cr_main_roll.set_trajectory(traj_dict_main)
prepare_st1 = flying_motor_cr_main_roll.prepare()

traj_dict_aux2 = {'positions': [pos_dict['motor_cr_aux2_roll'] - 1000 - 100,
                                pos_dict['motor_cr_aux2_roll'] + 1000 - 100], 'durations': [30]}
flying_motor_cr_aux2_roll.set_trajectory(traj_dict_aux2)
prepare_st2 = flying_motor_cr_aux2_roll.prepare()

traj_dict_aux3 = {'positions': [pos_dict['motor_cr_aux3_roll'] - 1000 + 100,
                                pos_dict['motor_cr_aux3_roll'] + 1000 + 100], 'durations': [30]}
flying_motor_cr_aux3_roll.set_trajectory(traj_dict_aux3)
prepare_st3 = flying_motor_cr_aux3_roll.prepare()

combine_status_list([prepare_st1, prepare_st2, prepare_st3]).wait()

st1 = flying_motor_cr_main_roll.kickoff()
st2 = flying_motor_cr_aux2_roll.kickoff()
st3 = flying_motor_cr_aux3_roll.kickoff()

data1 = np.array(flying_motor_cr_main_roll.data)
data2 = np.array(flying_motor_cr_aux2_roll.data)
data3 = np.array(flying_motor_cr_aux3_roll.data)
plt.figure(1, clear=True)
plt.plot(data1[:, 0], data1[:, 1], '.-')
plt.plot(data2[:, 0], data2[:, 1], '.-')
plt.plot(data3[:, 0], data3[:, 1], '.-')
# plt.plot(data[:, 2] - data[0, 2], data[:, 1], 'r.-')
# plt.plot(data[:, 0] - data[:, 2], 'k.-')


###
class FlyerForFlyableEpicsMotor(Device):
    def __init__(self, default_dets, motors, shutter, *args, **kwargs):
        super().__init__(parent=None, *args, **kwargs)

        # apb_stream_idx = dets.index(apb_stream)
        # self.apb_stream = dets[apb_stream_idx]

        self.default_dets = default_dets
        self.aux_dets = []
        self.dets = []
        self.motors = motors
        self.shutter = shutter
        self.complete_status = None

    def set_aux_dets(self, aux_dets):
        self.aux_dets = aux_dets

    def flush_dets(self):
        self.aux_dets = []
        self.dets = []

    def stage(self):
        print_to_gui(f'Preparing mono starting...', add_timestamp=True, tag='Flyer')
        motor_prep_status = combine_status_list([motor.prepare() for motor in self.motors])
        motor_prep_status.wait()
        print_to_gui(f'Preparing mono complete', add_timestamp=True, tag='Flyer')
        self.dets = self.default_dets + self.aux_dets
        print_to_gui(f'Fly scan staging starting...', add_timestamp=True, tag='Flyer')
        staged_list = super().stage()
        scan_duration = max([motor.current_trajectory_duration for motor in self.motors])
            # trajectory_manager.current_trajectory_duration
        for det in self.dets:
            if hasattr(det, 'prepare_to_fly'):
                det.prepare_to_fly(scan_duration)
            print_to_gui(f'{det.name} staging starting', add_timestamp=True, tag='Flyer')
            staged_list += det.stage()
        print_to_gui(f'Fly scan staging complete', add_timestamp=True, tag='Flyer')
        return staged_list

    def unstage(self):
        print_to_gui(f'Fly scan unstaging starting...', add_timestamp=True, tag='Flyer')
        unstaged_list = super().unstage()
        for det in self.dets:
            unstaged_list += det.unstage()
        self.flush_dets()
        print_to_gui(f'Fly scan unstaging complete', add_timestamp=True, tag='Flyer')
        return unstaged_list

    def kickoff(self):
        self.kickoff_status = DeviceStatus(self)
        self.complete_status = DeviceStatus(self)
        thread = threading.Thread(target=self.action_sequence, daemon=True)
        thread.start()
        return self.kickoff_status

    def action_sequence(self):

        print_to_gui(f'Detector kickoff starting...', add_timestamp=True, tag='Flyer')
        self.shutter.open(time_opening=True)
        det_kickoff_status = combine_status_list([det.kickoff() for det in self.dets])
        det_kickoff_status.wait()

        print_to_gui(f'Detector kickoff complete', add_timestamp=True, tag='Flyer')

        print_to_gui(f'Motor trajectory motion starting...', add_timestamp=True, tag='Flyer')

        self.motor_flying_status = combine_status_list([motor.kickoff() for motor in self.motors])
        self.kickoff_status.set_finished()

        self.motor_flying_status.wait()

        print_to_gui(f'Motor trajectory motion complete', add_timestamp=True, tag='Flyer')

        print_to_gui(f'Detector complete starting...', add_timestamp=True, tag='Flyer')
        det_complete_status = combine_status_list([det.complete() for det in self.dets])
        det_complete_status.wait()
        for motor in self.motors:
            motor.complete()
        self.shutter.close()

        print_to_gui(f'Detector complete complete', add_timestamp=True, tag='Flyer')
        self.complete_status.set_finished()

    def complete(self):
        # print(f'{ttime.ctime()} >>> COMPLETE: begin')
        if self.complete_status is None:
            raise RuntimeError("No collection in progress")
        return self.complete_status

    def describe_collect(self):
        return_dict = {}
        for det in self.dets:
            return_dict = {**return_dict, **det.describe_collect()}
        return return_dict

    def collect(self):
        # print_to_gui(f'{ttime.ctime()} Collect starting')
        print_to_gui(f'Collect starting...', add_timestamp=True, tag='Flyer')
        for det in self.dets:
            yield from det.collect()
        # print_to_gui(f'{ttime.ctime()} Collect finished')
        print_to_gui(f'Collect complete', add_timestamp=True, tag='Flyer')

    def collect_asset_docs(self):
        print_to_gui(f'Collect asset docs starting...', add_timestamp=True, tag='Flyer')
        for det in self.dets:
            yield from det.collect_asset_docs()
        print_to_gui(f'Collect asset docs complete', add_timestamp=True, tag='Flyer')


# flyer_apb = FlyerHHM([apb_stream, pb9.enc1, xs_stream], hhm, shutter, name='flyer_apb')
# flyer_apb = FlyerHHM([apb_stream, pb9.enc1], hhm, shutter, name='flyer_apb')
flyer_johann_rolls = FlyerForFlyableEpicsMotor([apb_stream],
                                               [flying_motor_cr_main_roll,
                                                flying_motor_cr_aux2_roll,
                                                flying_motor_cr_aux3_roll],
                                               shutter, name='flyer_apb')

def fly_johann_rolls_plan(name=None, comment=None, trajectory_dictionary=None,
                          element='', e0=0, line='', metadata={}):

    flying_motor_cr_main_roll.set_trajectory(trajectory_dictionary['main'])
    flying_motor_cr_aux2_roll.set_trajectory(trajectory_dictionary['aux2'])
    flying_motor_cr_aux3_roll.set_trajectory(trajectory_dictionary['aux3'])

    detectors = ['Pilatus 100k']
    aux_detectors = get_detector_device_list(detectors, flying=True)
    flyer_johann_rolls.set_aux_dets(aux_detectors)
    detectors_dict = {k :{'device' : v} for k, v in zip(detectors, aux_detectors)}
    md_general = get_scan_md(name, comment, detectors_dict, '.dat')

    md_scan = {'experiment': 'fly_scan',
               'spectrometer': 'johann',
               'spectrometer_config': rowland_circle.config,
               'spectrometer_trajectory_dictionary': trajectory_dictionary,
               'element': element,
               'line': line,
               'e0': e0}
    md = {**md_scan, **md_general, **metadata}

    @bpp.stage_decorator([flyer_johann_rolls])
    def _fly(md):
        plan = bp.fly([flyer_johann_rolls], md=md)
        yield from monitor_during_wrapper(plan, [flying_motor_cr_main_roll.motor.user_readback,
                                                 flying_motor_cr_aux2_roll.motor.user_readback,
                                                 flying_motor_cr_aux3_roll.motor.user_readback])
    yield from _fly(md)


def pseudo_fly_scan_johann_xes_plan(name=None, comment=None, detectors=[],
                              mono_energy=None, mono_angle_offset=None,
                              central_emission_energy=None,
                              trajectory_dictionary=None,
                              element='', line='', e0=None,
                              metadata={}):

    if mono_angle_offset is not None: hhm.set_new_angle_offset(mono_angle_offset)
    metadata = {**metadata, **{'spectrometer_central_energy': central_emission_energy}}
    yield from bps.mv(hhm.energy, mono_energy)
    pos_dict = johann_emission._forward({'energy': central_emission_energy})
    johann_emission.motor_cr_main_roll.move(pos_dict['motor_cr_main_roll'])
    johann_emission.motor_cr_aux2_roll.move(pos_dict['motor_cr_aux2_roll'])
    johann_emission.motor_cr_aux3_roll.move(pos_dict['motor_cr_aux3_roll'])
    yield from prepare_johann_scan_plan(detectors, central_emission_energy)
    yield from fly_johann_rolls_plan(name=name, comment=comment, trajectory_dictionary=trajectory_dictionary,
                                     element=element, line=line, e0=e0, metadata=metadata)
    # yield from general_energy_step_scan(all_detectors, johann_emission, emission_energy_list, emission_time_list, md=md)


trajectory_dictionary = {'main': traj_dict_main,
                         'aux2': traj_dict_aux2,
                         'aux3': traj_dict_aux3}

RE(pseudo_fly_scan_johann_xes_plan(name='test', comment='', detectors=['Pilatus 100k'],
                                mono_energy=7200, mono_angle_offset=None,
                                central_emission_energy=7085,
                                trajectory_dictionary=trajectory_dictionary,
                                element='Fe', line='Kb', e0=7059,
                                metadata={}))



for j in range(25):
    RE(bps.mvr(johann_det_arm.motor_det_th1, 3))
    RE(bps.sleep(2))
    RE(bps.mvr(johann_det_arm.motor_det_th1, -3))




#########

# johann_x = [#-8.55,
#             -6.058, -3.55, -1.057, 1.55, 3.944, 6.445]
# uids = [
#    # '4dace7fc-994f-4da4-a2c2-7df3e7f1c233',
#     'cd5ffdf6-6723-45fe-8fd3-39dfdb703f3c',
#     '0f568080-b2a0-4e33-a766-c4ac4638178d',
#     '979a380a-167c-432f-acc6-36bceeea1c6d',
#     'cd067a65-e402-44f1-9bcd-8f10c01c51d2',
#     '96dd4ad7-f178-42b2-90f7-3b37978944b9',
# '32d3bd90-ab2d-40f4-bcb3-62173486c90a'
# ]

johann_x = [-7.5]

uids = ['1c1d309c-9dc2-4d06-8f71-8c74d1575f17', ]


fwhm = []
plt.figure(1, clear=True)
for uid in uids:
    _fwhm = estimate_peak_fwhm_from_roll_scan(db, uid, plotting=True, fignum=1, clear=False)
    fwhm.append(_fwhm)

plt.figure(2, clear=True)
plt.plot(johann_x, fwhm, 'k.-')


##

# uids = ('ed7aeea7-e50d-4f1d-828f-f335383210cc',
#  '5ee2f655-c691-400b-8cc0-8f5e5a57aa71',
#  '40a9f077-7f82-4d40-946f-98a2b1030355',
#  '4f5dac85-bc61-4757-b868-900b221d2dab',
#  '75612f5d-0d25-45c6-ba9f-41c558e8a979',
#  '464e994e-ca73-4ac6-a7a8-52d07872a020',
#  '398398c7-5d44-4e6b-b4d0-3c93cf95bb2d',
#  'd3776097-5a61-457e-9318-a447d17400ff',
#  '0d60c03a-b75b-418f-bb94-02bdfe3f39cb',
# '0e6f37da-8d43-4bf5-935d-8fc348a35bc6',
# 'd8278aa1-21d4-424a-b00b-2223e6c3ac42',
# 'ecacc8a6-ecb7-483d-8cfa-e097839dceae'
#         )
# tweak_motor = johann_spectrometer_x.name
# scan_motor = 'johann_main_crystal_motor_cr_main_roll'
#
# uids = (#'135e21d6-f6bc-439c-8840-64e3c769fab7',
#  'c8b29993-8a05-475d-ab0e-78965ecd4617',
#  'b2257dcc-7433-4445-ab1b-fb494370edee',
#  '775c8aaa-dcac-49eb-bf2a-dc3946423ba4',
#  '38322180-d2c7-469c-83c0-2c6b623f8734',
#  '677d3d31-9808-44b2-8cea-9c9683ead1fd',
#  'fa9b925a-4f39-456f-9b92-fcfec349b13d',
#  '29a90988-23cb-4186-9596-67c184e016cb',
#  '52d54744-d2d2-4a0d-a748-d01ff3b9fc6e')
#
# tweak_motor = johann_aux2_crystal.motor_cr_aux2_x.name
# scan_motor = 'johann_aux2_crystal_motor_cr_aux2_roll'

uids = ('886406a3-b684-4a42-a62b-3a831fcba65e',
 '52643015-84ff-4962-9bd1-0407fbe81830',
 'a8d60e8b-2ed3-4127-9eb9-2d4b43d42f0a',
 '51bf692a-cf50-49e4-9eb9-222ddc678ca4',
 'be00338e-c124-41de-9074-220d80bbda37',
 '5290b704-f7fc-4dcc-a8e0-cf54ec4fdfad',
 'fcf783e6-bea4-48b1-8497-66a250630d2b',
 'db72c465-a47b-44ae-bd84-74c5fd4a59a3',
 '13eee9db-f5a2-4dc3-974a-5c91090a4681')

tweak_motor = johann_aux3_crystal.motor_cr_aux3_x.name
scan_motor = 'johann_aux3_crystal_motor_cr_aux3_roll'

fwhm = []
motor_pos = []
plt.figure(1, clear=True)
for uid in uids:
    hdr = db[uid]
    df = hdr.table()
    _fwhm = _estimate_peak_fwhm_from_roll_scan(df, scan_motor, 'pil100k_stats1_total', plotting=True, fignum=1, clear=False)
    fwhm.append(_fwhm)
    motor_pos.append(hdr.start[tweak_motor])


plt.figure(2, clear=True)
plt.plot(motor_pos, fwhm, 'k.-')



# uids = ['97210e1b-d8e0-436c-97e6-8284d10ce1a4', # main
        # ]

t_main = db['97210e1b-d8e0-436c-97e6-8284d10ce1a4'].table()
t_aux2 = db['4626bc5e-b945-4fca-80e0-af9bda173d7c'].table()
t_aux3 = db['a82eaf26-2281-47e4-b601-815fbd489060'].table()

def plot_plot(t, x_key, y_key):
    x = t[x_key]
    y = t[y_key]
    y_norm = (y - np.mean(y[-5:])) / (y.max() - np.mean(y[-5:]))
    plt.plot(x - np.median(x), y_norm)


plt.figure(1, clear=True)
plot_plot(t_main, 'johann_main_crystal_motor_cr_main_roll', 'pil100k_stats1_total')
plot_plot(t_aux2, 'johann_aux2_crystal_motor_cr_aux2_roll', 'pil100k_stats1_total')
plot_plot(t_aux3, 'johann_aux3_crystal_motor_cr_aux3_roll', 'pil100k_stats1_total')

# plt.plot(t_aux2.johann_aux2_crystal_motor_cr_aux2_roll, t_aux2.pil100k_stats1_total )
# plt.plot(t_aux3.johann_aux3_crystal_motor_cr_aux3_roll, t_aux3.pil100k_stats1_total )

# '68f0f134-fcda-4b15-9053-e47031659a18' # main bragg
 # aux2

def plot_bragg_data(uid, x_key, y_key, n1=0, n2=3):
    t = db[uid].table()
    x = t[x_key]
    y = t[y_key]
    y_norm = (y - np.mean(y[n1:n2])) / (y.max() - np.mean(y[n1:n2]))
    plt.plot(x, y_norm)

plt.figure(1, clear=True)
plot_bragg_data('68f0f134-fcda-4b15-9053-e47031659a18', 'johann_main_crystal_bragg', 'pil100k_stats1_total')
plot_bragg_data('792720f5-89c8-45c7-8b08-e3f1a3b941cf', 'johann_aux2_crystal_bragg', 'pil100k_stats1_total')
plot_bragg_data('20da49aa-5019-4470-9fe8-5d9feeb4ae4c', 'johann_aux3_crystal_bragg', 'pil100k_stats1_total', n1=-3, n2=-1)



# main crystal
uids = \
('37e16188-03e6-4be1-a8ab-d8217cea1731',
 'b594b1c0-d82a-405d-b6f9-02f203dc337b',
 '60f686cb-86fb-4072-b53b-52faf79bdf23',
 'e9500360-e130-4466-bb6a-d9b6005f5238',
 '38a05189-9289-45f1-86cb-2de503215029',
 '256b1c78-d43e-4b80-b87a-0521a80caef6',
 'dabbb337-b2ed-4d97-bbee-ed82eaf97acc',
 'f5c063b2-7c48-40af-85c5-b07929d993d3',
 'f54fd53e-9ca5-4bfc-96f5-1dd33cd32336',
 '3eae971a-97ec-4832-bc46-435b369a5b9e',
 '85f6a92c-324e-4be5-a9f9-804dd8bb1d67',
 '93d30b7d-09e3-4b3b-996c-918fb9967d0a',
 'a7e635d1-96cb-49d4-8832-f16ac62c9b0b',
 '549d7e0a-6f55-475f-8f10-4364bab7667b',
 '357de376-e16b-4fea-854b-1d20d1aa4d66'
 )

uids = \
('28f4e4d0-0000-45e4-9440-fc48731dfe05',
 #'65efae59-e92b-47e7-9444-40f2666d8f0c',
 'ea9dbfd0-2134-479c-8c00-ab50cd8696e7',
 '251642dd-ca9c-4f6b-838e-0f4263bc56ec',
 'a80ef314-9569-4027-8c12-fbbb8011a6ee',
 '09e9ba0f-20a9-436d-ae85-6e6c153446b8',
 'e873ced5-259c-460c-b465-8d52798ce3f8',
 '0883a0d0-9bd3-4332-b1d1-bc1e6c2f8039',
 '1388d51a-41b2-40ca-b8d2-b906e3eb3062')


tweak_motor = johann_spectrometer_x.name
scan_motor = 'johann_main_crystal_motor_cr_main_roll'



uids = \
(#'5a229f07-93b7-4b6c-aac9-fea745d056d3',
 'c0afe15e-8a0d-4d52-b70d-613f1937415f',
 'b3d35ed9-0c6e-42e0-b665-261c9ea36803',
 'b0ea60cc-5fb9-4436-a1c7-59c2921a69df',
 'ec9e5f00-4af3-4a74-913b-8760e56ad72a',
 'af9b9cb6-f83b-460b-b6a2-b87b0fe1814a',
 '7a2e6a34-82d5-44b9-866b-f5babe757d67',
 'ad5dbb20-28ef-46fd-bf51-496e96bab9dc',
 '9c5af685-d1e8-4a48-86cd-62900be8c7aa')




tweak_motor = johann_aux2_crystal.motor_cr_aux2_x.name
scan_motor = 'johann_aux2_crystal_motor_cr_aux2_roll'

uids = ('12399c12-6e48-43b5-a26a-491259cbda03',
 '0bab4c15-3fc4-4fc3-84b6-89134479443c',
 '11cd518f-e4ca-43da-b3d4-1ff0b0ff6ed6',
 '1c627016-5a3d-4257-a0c8-5b819abcd2c3',
 'e38b8b31-73d7-4af8-9781-8dcb553e2b35',
 'ebc9f3bc-7c27-49c5-a8d4-5d1da312b950',
 '20ba24b0-b9bf-4f6c-bc75-aaad07def7b5',
 '55510315-d5f9-453c-8074-ab22e380122a',
 '085ac43e-5ffb-437b-88db-5d8fbe4bcf44',
 'b60da7c8-cb01-4d16-937e-99c9ea5d367c',
 '65794f6d-5d85-4b8a-b909-83c20911bf3f',
 )

uids = \
(#'8b97d03c-a1a3-494c-ab30-2a96e70ea331',
 '89d74def-5247-4825-8cdd-339a20552ca8',
 'e86769d8-ba26-420c-a435-76f0342fd3c3',
 #'488056dd-a9a3-4051-959d-2600e6fda66d',
 '9ac8aab4-6520-403e-82ce-8a31f861e386',
 'c6fdffbc-ea86-418f-8d40-f11e3cc96152',
 'f558b42c-8131-4c9f-b856-820b35f4ee8a',
 'b2d2e157-468d-4105-973c-c639f0c31507',
 'd4a71e47-45b7-4ca5-a2d3-3983ef8420a4',
 '59f702bf-d26c-4c58-869e-a63c0e91a9bf',
 '2afd4b9d-94fe-43b3-924e-896c247d8666')

uids = \
('53760d42-63d2-47ff-84c5-de111cbbb136',
 'e1d69459-8e4c-45b5-83b8-8306a2fa7107',
 'e6f6deef-848b-47ff-8414-a0755916e8f5',
 'fc2040af-af1a-465a-97ba-1567cc41b785',
 '9b7ac035-d97c-460d-8d7b-ff541aa08738',
 '5e3919ef-13d2-4a97-8ebd-2bb870bd93fd',
 '5066fde0-df7e-433e-995f-7bb256870675',
 '39a6639c-cc2d-4df8-bb27-f16b6fafe431',
 '928b333e-855d-48f2-95b1-c954cfc25919')


tweak_motor = johann_aux3_crystal.motor_cr_aux3_x.name
scan_motor = 'johann_aux3_crystal_motor_cr_aux3_roll'


fwhm = []
motor_pos = []
plt.figure(1, clear=True)
for uid in uids:
    _fwhm = estimate_peak_fwhm_from_roll_scan(db, uid, x_col=scan_motor, y_col='pil100k_stats1_total', plotting=True, fignum=1, clear=False)
    # _fwhm = estimate_peak_intensity_from_roll_scan(db, uid, x_col=scan_motor, y_col='pil100k_stats1_total', plotting=True, fignum=1, clear=False)
    fwhm.append(_fwhm)
    hdr = db[uid]
    motor_pos.append(hdr.start[tweak_motor])
    # plot_roll_scan(db, uid, x_col=scan_motor, y_col='pil100k_stats1_total')

motor_pos = np.array(motor_pos).squeeze()
fwhm = np.array(fwhm).squeeze()

from numpy.polynomial import Polynomial as P
p = P.fit(motor_pos, fwhm, 3)

fwhm_fit = p(np.array(motor_pos))
plt.figure(2, clear=True)
plt.plot(motor_pos, fwhm, 'k.-')
plt.plot(motor_pos, fwhm_fit, 'r-')

def _get_minimum_position(x, y, deg=3):
    x = np.array(x)
    y = np.array(y)
    p = P.fit(x, y, deg)
    x_extrema = p.deriv().roots()
    x_minima = x_extrema[p.deriv().deriv()(x_extrema) > 0]
    x_minima = x_minima[(x_minima>=x.min()) & (x_minima<=x.max())]
    if len(x_minima) == 0:
        x_minimum = x[np.argmin(y)]
    else:
        y_minima = p(x_minima)
        x_minimum = x_minima[np.argmin(y_minima)]

    plt.figure(2, clear=True)
    plt.plot(x, y, 'k.-')
    plt.plot(x, p(x), 'r-')
    plt.vlines([x_minimum, x_minimum], y.min(), y.max(), colors='m')

    return x_minimum

_get_minimum_position(motor_pos, fwhm)

from xas.db_io import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, load_pil100k_dataset_from_db



uid = 358337

apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, uid)
raw_dict = translate_apb_dataset(apb_df, energy_df, energy_offset)

apb_trigger_pil100k_timestamps = load_apb_trig_dataset_from_db(db, uid, use_fall=True,
                                                               stream_name='apb_trigger_pil100k')
pil100k_dict = load_pil100k_dataset_from_db(db, uid, apb_trigger_pil100k_timestamps)
raw_dict = {**raw_dict, **pil100k_dict}

hdr = db[uid]

run_quick_alignment_scans_for_crystal(motor_description='Johann Main Crystal Roll', scan_range=800, velocity=25, tweak_motor=johann_spectrometer_x, tweak_motor_rel_start=-10, tweak_motor_rel_stop=10, tweak_motor_num_steps=9)
run_quick_alignment_scans_for_crystal(motor_description='Johann Aux2 Crystal Roll', scan_range=800, velocity=25, tweak_motor=johann_aux2_crystal.motor_cr_aux2_x, tweak_motor_rel_start=-10000, tweak_motor_rel_stop=10000, tweak_motor_num_steps=9)
run_quick_alignment_scans_for_crystal(motor_description='Johann Aux3 Crystal Roll', scan_range=800, velocity=25, tweak_motor=johann_aux3_crystal.motor_cr_aux3_x, tweak_motor_rel_start=-10000, tweak_motor_rel_stop=10000, tweak_motor_num_steps=11)




###
detectors = ['Pilatus 100k']
motor_dict = {'main': johann_emission.motor_cr_main_roll}

roll_center = 1000
roll_delta = 500

trajectory_dict = {'main': {'positions': [roll_center - roll_delta, roll_center + roll_delta],
                            'durations': [10]}}

_plan = general_epics_motor_fly_scan(detectors, motor_dict, trajectory_dict, {})
RE(_plan)

from xas.db_io import load_apb_trig_dataset_from_db, load_pil100k_dataset_from_db

uid = '0cfd6579-6010-4dbe-a41d-10d2a45a5385'


uid = -2

hdr = db[uid]
apb_trigger_pil100k_timestamps = load_apb_trig_dataset_from_db(db, uid, use_fall=True,
                                                               stream_name='apb_trigger_pil100k')
pil100k_dict = load_pil100k_dataset_from_db(db, uid, apb_trigger_pil100k_timestamps)

roll_monitor_df = hdr.table('johann_main_crystal_motor_cr_main_roll_monitor')

roll_monitor_df['timestamp'] = (roll_monitor_df.time.values - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


plt.figure(2, clear=True)
plt.plot(roll_monitor_df.timestamp.values, roll_monitor_df.johann_main_crystal_motor_cr_main_roll, '.-')
plt.plot(apb_trigger_pil100k_timestamps, 1000*np.ones(apb_trigger_pil100k_timestamps.shape), '.-')



###


_crystal_alignment_dict = {'main': {'roll': 'Johann Main Crystal Roll',
                                    'yaw':  'Johann Main Crystal Yaw',
                                    'x':    'Johann Crystal Assy X'},
                           'aux2': {'roll': 'Johann Aux2 Crystal Roll',
                                    'yaw':  'Johann Aux2 Crystal Yaw',
                                    'x':    'Johann Aux2 Crystal X'},
                           'aux3': {'roll': 'Johann Aux3 Crystal Roll',
                                    'yaw':  'Johann Aux3 Crystal Yaw',
                                    'x':    'Johann Aux3 Crystal X'}}




def quick_crystal_piezo_scan(crystal=None, axis=None, scan_range=None, velocity=None, pil100k_exosure_time=0.1, plot_func=None, liveplot_kwargs=None, md=None):
    motor_description = _crystal_alignment_dict[crystal][axis]
    motor_device = get_motor_device(motor_description, based_on='description')
    detectors = [apb.ch1, pil100k.stats1.total, pil100k.stats2.total, pil100k.stats3.total, pil100k.stats4.total, ]

    print_to_gui(f'Quick scanning motor {motor_description}', tag='Spectrometer')

    num_images = (scan_range / velocity  + 1) / pil100k_exosure_time
    print(num_images)
    pil100k_init_exposure_time = pil100k.cam.acquire_period.get()
    pil100k_init_num_images = pil100k.cam.num_images.get()
    pil100k_init_image_mode = pil100k.cam.image_mode.get()

    pil100k.set_exposure_time(pil100k_exosure_time)
    pil100k.set_num_images(num_images)

    pil100k.cam.image_mode.set(1).wait()

    start_acquiring_plan = bps.mv(pil100k.cam.acquire, 1)
    yield from ramp_motor_scan(motor_device, detectors, scan_range, velocity=velocity, return_motor_to_initial_position=True, start_acquiring_plan=start_acquiring_plan, md=md)

    pil100k.set_exposure_time(pil100k_init_exposure_time)
    pil100k.set_num_images(pil100k_init_num_images)
    pil100k.cam.image_mode.set(pil100k_init_image_mode).wait()


def quick_crystal_piezo_tune(**kwargs):
    yield from quick_crystal_piezo_scan(**kwargs)
    com = obtain_ramp_scan_com_plan(db, -1)
    motor_description = _crystal_alignment_dict[kwargs['crystal']][kwargs['axis']]
    yield from move_motor_plan(motor_attr=motor_description, based_on='description', position=com)

def get_tweak_motor_positions_for_crystal(crystal, motor_range_mm, motor_num_steps):
    motor_description = _crystal_alignment_dict[crystal]['x']
    motor_obj = get_motor_device(motor_description, based_on='description')
    motor_pos_init = motor_obj.position

    motor_pos_start = motor_pos_init - motor_range_mm / 2
    motor_pos_stop = motor_pos_init + motor_range_mm / 2

    motor_pos_steps = np.linspace(motor_pos_start, motor_pos_stop, motor_num_steps)

    motor_low_lim = motor_obj.low_limit # check this
    motor_high_lim = motor_obj.high_limit  # check this
    motor_pos_steps = motor_pos_steps[(motor_pos_steps >= motor_low_lim) & (motor_pos_steps <= motor_high_lim)]
    return motor_pos_init, motor_pos_steps, motor_description


def estimate_peak_fwhm_from_quick_roll_scan(db, uid, x_col='johann_main_crystal_motor_cr_main_roll',
                                      y_col='pil100k_stats1_total', **kwargs):
    df = process_monitor_scan(db, uid, det_for_time_base='pil100k')
    df = df[3: df.shape[0] - 3]
    return _estimate_peak_fwhm_from_roll_scan(df, x_col, y_col, **kwargs)[0]

def process_quick_crystal_piezo_roll_scan(crystal=None, pil100k_roi_num=None, alignment_data=None):
    motor_description = _crystal_alignment_dict[crystal]['roll']
    motor_device = get_motor_device(motor_description, based_on='description')
    x_col = motor_device.name
    y_col = f'pil100k_stats{pil100k_roi_num}_total'
    fwhm = estimate_peak_fwhm_from_quick_roll_scan(db, -1, x_col=x_col, y_col=y_col)
    start = db[-1].start
    uid = start.uid
    _dict = {'uid': uid,
             'fwhm': fwhm,
             'tweak_motor_description':     start['tweak_motor_description'],
             'tweak_motor_position':        start['tweak_motor_position']}
    alignment_data.append(_dict)


def run_alignment_scans_for_crystal_bundle(crystal=None, alignment_by=None, pil100k_roi_num=None,
                                           alignment_data=None,
                                           scan_range_roll=None, scan_range_yaw=None, velocity=None,
                                           tweak_motor_range=None, tweak_motor_num_steps=None,
                                           plot_func=None, liveplot_kwargs=None):
    if alignment_data is None:
        alignment_data = []
    tweak_motor_init_pos, tweak_motor_pos, tweak_motor_description = get_tweak_motor_positions_for_crystal(crystal, tweak_motor_range, tweak_motor_num_steps)

    plans = []

    for i, _pos in enumerate(tweak_motor_pos):
        plans.append({'plan_name': 'print_message_plan',
                      'plan_kwargs': {'msg': f'Aligning motor {tweak_motor_description} (step {i + 1}, position={_pos})',
                                      'add_timestamp': True,
                                      'tag': 'Spectrometer'}})
        plans.append({'plan_name': 'move_motor_plan',
                      'plan_kwargs': {'motor_attr': tweak_motor_description,
                                      'based_on': 'description',
                                      'position': _pos}})
        if crystal != 'main':
            plans.append({'plan_name': 'quick_crystal_piezo_tune',
                          'plan_kwargs': {'crystal': crystal,
                                          'axis': 'yaw',
                                          'scan_range': scan_range_yaw,
                                          'velocity': velocity,
                                          'plot_func': plot_func,
                                          'liveplot_kwargs': liveplot_kwargs}})

        md = {'tweak_motor_description': tweak_motor_description,
              'tweak_motor_position': _pos}

        if alignment_by == 'emission':
            plans.append({'plan_name': 'quick_crystal_piezo_scan',
                          'plan_kwargs': {'crystal': crystal,
                                          'axis': 'roll',
                                          'scan_range': scan_range_roll,
                                          'velocity': velocity,
                                          'plot_func': plot_func,
                                          'liveplot_kwargs': liveplot_kwargs,
                                          'md': md}})
            plans.append({'plan_name': 'process_quick_crystal_piezo_roll_scan',
                          'plan_kwargs': {'crystal': crystal,
                                          'pil100k_roi_num': pil100k_roi_num,
                                          'alignment_data': alignment_data}})


        elif alignment_by == 'elastic':
            pass

    plans.append({'plan_name': 'move_motor_plan',
                  'plan_kwargs': {'motor_attr': tweak_motor_description,
                                  'based_on': 'description',
                                  'position': tweak_motor_init_pos}})

    return plans



















uids = (
'2c0ca392-00fe-4cc8-b4d5-eecf07f41547',
'2a4a899b-0a1b-4962-a7bf-92542026d1d8',
'fe009829-41ac-49b9-b2e7-ba2e91430a05',
'4148c7dd-fb0b-458f-8553-73a088124198',
'b8abe0c6-4a0a-422e-aacf-1a80e1fe0bc9',
'863e5780-bce0-4eb1-9145-fc1dce9e64bc',
'cca6e4e6-54a8-4690-98fa-076cfb3ab4d8',
'a0a2dc82-de59-4b07-a22d-84cae2e859d1',
'6790bbc8-cd29-45de-b825-0c76dcbec351',
'62ab5550-370b-4877-b200-6910a793e031',
'e1373193-cc3c-4376-9df5-7a26ddf19b62',
'd7f87aae-4664-4eca-b14d-99a1d6965ee4',
'fe9a725b-282b-4f44-a6a8-c93acd4d48bb',
'7de32b40-be34-491d-bc18-f22f1c33165a',
'f37ba8e4-3466-400c-a291-e1caa5be3037',
'a55e6c90-9329-417d-9288-fed688be12c9',
'40fdbc4b-0fb0-4f27-b2ef-4b3de4e5a689',
'7c4b889a-ecfa-4bdb-bbdf-7974644c1bf1',
)

ALIGNMENT_DATA = []
for uid in uids:
    hdr = db[uid]
    if 'tweak_motor_description' in hdr.start:
        RE(process_crystal_piezo_roll_scan(crystal='aux2', pil100k_roi_num=1, alignment_data=ALIGNMENT_DATA, uid=uid))




[{'uid': '2a4a899b-0a1b-4962-a7bf-92542026d1d8',
  'fwhm': 109.85048884796174,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -8674.124},
 {'uid': '4148c7dd-fb0b-458f-8553-73a088124198',
  'fwhm': 106.86883456590192,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -7174.124},
 {'uid': '863e5780-bce0-4eb1-9145-fc1dce9e64bc',
  'fwhm': 103.67161694659023,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -5674.124},
 {'uid': 'a0a2dc82-de59-4b07-a22d-84cae2e859d1',
  'fwhm': 101.33901125348939,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -4174.124},
 {'uid': '62ab5550-370b-4877-b200-6910a793e031',
  'fwhm': 98.63497076951717,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -2674.124},
 {'uid': 'd7f87aae-4664-4eca-b14d-99a1d6965ee4',
  'fwhm': 96.96312591408594,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -1174.1239999999998},
 {'uid': '7de32b40-be34-491d-bc18-f22f1c33165a',
  'fwhm': 95.9164672340288,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 325.8760000000002},
 {'uid': 'a55e6c90-9329-417d-9288-fed688be12c9',
  'fwhm': 94.67777548320714,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 1825.8760000000002},
 {'uid': '7c4b889a-ecfa-4bdb-bbdf-7974644c1bf1',
  'fwhm': 93.88133496961007,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 3325.8759999999997}]



### main crystal elastic on Cu foil
[{'uid': '56e31fc3-cdbd-4f98-81c4-8511a8dcf7b1',
  'fwhm': 1.4969813851357685,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 972.888154149},
 {'uid': '7ae0f696-83ee-4c22-b443-3d3cf41b4c4c',
  'fwhm': 1.4849839817607062,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 975.388154149},
 {'uid': 'c8b1c8e3-5e67-489b-8b2d-97ac8ab7df61',
  'fwhm': 1.5552265480819187,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 977.888154149},
 {'uid': '4be2883b-54f4-48b2-865f-3e0c6980a4a4',
  'fwhm': 1.6928793954039065,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 980.388154149},
 {'uid': '0bb8f882-d7ff-474e-a90c-ca1f421cbff1',
  'fwhm': 1.649486713891747,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 982.888154149},
 {'uid': '1c052e4c-afcd-4a3e-921d-c7a31f8ef8f2',
  'fwhm': 1.7126857458733866,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 985.388154149},
 {'uid': 'b36be526-92c7-46f1-b9c1-22c3b7228408',
  'fwhm': 1.7495045014229618,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 987.888154149},
 {'uid': '21647db6-4acc-4886-87ad-fc9d3f8af841',
  'fwhm': 1.8407076891280667,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 990.388154149},
 {'uid': '3b69e77d-5b56-4920-8997-01e3a7f3af1f',
  'fwhm': 1.8300737904373818,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 992.888154149}]

# main crystal fluorescence on Cu foil
[{'uid': 'b01d5359-8832-4a8b-b42c-0434477c5c3b',
  'fwhm': 103.0131452325918,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 972.888154149},
 {'uid': 'c5c3fb6b-46aa-456c-817d-92de2e7b0bc5',
  'fwhm': 101.25272973337042,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 975.388154149},
 {'uid': 'f12a8319-e51b-462d-bdb5-8e5134d35c3c',
  'fwhm': 100.85851172676712,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 977.888154149},
 {'uid': '0da229dc-7721-40fd-8bf5-6a27caaea248',
  'fwhm': 99.20244253792407,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 980.388154149},
 {'uid': '17881529-2ccb-4cf0-a3d2-2dd1626d7ba6',
  'fwhm': 99.36881768443118,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 982.888154149},
 {'uid': 'e54f6a69-e30e-4bfd-b6ae-1210eefbcfff',
  'fwhm': 101.23082158479315,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 985.388154149},
 {'uid': 'c21f5926-b682-47ad-95bf-c6daffa43eda',
  'fwhm': 103.16026433967136,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 987.888154149},
 {'uid': '67bdeada-a861-4560-84a5-1ff0dc741720',
  'fwhm': 106.10226721603112,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 990.388154149},
 {'uid': '3189f05e-0a7a-40cc-a076-c3c0edc0dc68',
  'fwhm': 109.74666410627765,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 992.888154149}]


# main crystal elastic peak on water
[{'uid': '21dd2097-f3c5-42c1-837a-26bf2a57be82',
  'fwhm': 1.5203829584625055,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 972.888154149},
 {'uid': 'e9338e5f-462f-4b74-bade-48d062e8a32d',
  'fwhm': 1.5598377745800462,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 975.388154149},
 {'uid': '2ad1e27e-8e0a-4928-8cde-4abe021e47c4',
  'fwhm': 1.530486475104226,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 977.888154149},
 {'uid': '1d06885a-c17b-4d66-af4f-c76a456b5847',
  'fwhm': 1.6073704188665943,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 980.388154149},
 {'uid': '99b20bd8-3de5-40de-afdd-a239fbdd7d54',
  'fwhm': 1.6830465732546145,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 982.888154149},
 {'uid': 'c2568092-3eaf-40f8-82f5-d63cf5d9754f',
  'fwhm': 1.6799933046841034,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 985.388154149},
 {'uid': 'a49dd39d-0eb2-46f8-85fe-747cfc5901f7',
  'fwhm': 1.7063572256956832,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 987.888154149},
 {'uid': 'aed14f3e-e339-49ef-b8df-1d98e45bdd46',
  'fwhm': 1.8170261371833476,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 990.388154149},
 {'uid': '716c8f9a-39a9-43e6-9d2b-791c6c9c20d1',
  'fwhm': 1.8061672367330175,
  'tweak_motor_description': 'Johann Crystal Assy X',
  'tweak_motor_position': 992.888154149}]


# aux2 crystal elastic peak on water
[{'uid': 'd2d45639-7787-4f06-8c5f-97daa0d16cac',
  'fwhm': 2.00445572829085,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -12673.993},
 {'uid': '12fba8bf-2a38-4b0b-9330-39014a32328e',
  'fwhm': 1.9520583201510817,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -10173.993},
 {'uid': '01081b77-dca2-40bb-85f9-f947306ac094',
  'fwhm': 1.8096046514519912,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -7673.993},
 {'uid': 'e41d6da7-01c2-47dc-9b5f-01ce54b28519',
  'fwhm': 1.7827555236535773,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -5173.993},
 {'uid': '1ddb9cb1-26c6-4d59-8561-6c632eef82a3',
  'fwhm': 1.6361589040898252,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -2673.9930000000004},
 {'uid': '04a1903c-8adf-40af-87f0-d49a9016344c',
  'fwhm': 1.5704819213387964,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -173.9930000000004},
 {'uid': '346b13b3-9320-45bf-8a48-112701f68367',
  'fwhm': 1.5424939347876716,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 2326.0069999999996},
 {'uid': 'bb93219c-5081-4835-9a00-20ea3e6f74ef',
  'fwhm': 1.4700413649352413,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 4826.007},
 {'uid': 'faa72d77-a05f-4ec6-a0f6-d162dbbc9968',
  'fwhm': 1.4617302860033305,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 7326.007},
 {'uid': 'c5d51498-4c55-45e4-9a8e-c2eee2cb9e02',
  'fwhm': 1.368008458037366,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 8000.0},
 {'uid': 'e3214fc5-021d-4f12-b4f5-20e6ddc9dabb',
  'fwhm': 1.4516299726528814,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 10000.0},
 {'uid': 'b4af7372-01e4-401a-9edc-62724869131d',
  'fwhm': 1.644220724688239,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 12000.0}]

#aux2 crystal emission on foil
[{'uid': '70d2c1e9-9f05-4723-9900-f505beb70720',
  'fwhm': 125.37645892598186,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -12300.0},
 {'uid': 'c727947e-30f1-4cdc-8d25-77b603a74aa7',
  'fwhm': 116.9006887461228,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -9900.0},
 {'uid': '5b4ea25b-cb18-4884-87c8-fc49ef0848cd',
  'fwhm': 111.42263227664591,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -7500.0},
 {'uid': '316e8cc7-823e-4207-a65f-6a1ee7fe0f1a',
  'fwhm': 105.6021815665315,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -5100.0},
 {'uid': '06de6e90-8bd4-47b8-bc2b-09f00fa0e42f',
  'fwhm': 101.11294992660271,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -2700.0},
 {'uid': 'ec28477c-f5c7-45bf-9142-9feff85ae96b',
  'fwhm': 97.74861285840143,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -300.0},
 {'uid': 'd451ea35-5203-4a49-94af-04c78cc3d96b',
  'fwhm': 95.71593851942123,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 2100.0},
 {'uid': 'ea82a574-e42d-4e62-9409-efc2603849a7',
  'fwhm': 94.40073361374868,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 4500.0},
 {'uid': '32b863eb-3a4f-4f60-85e0-45e63a03f023',
  'fwhm': 94.53097367988698,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 6900.0},
 {'uid': '18cf624b-dabe-41c2-a074-1310b7592aed',
  'fwhm': 95.42932443648715,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 9300.0},
 {'uid': 'eb87ab7c-b2c2-4531-afc9-f9c45e279ae4',
  'fwhm': 98.01414694788969,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 11700.0}]

# aux2 crystal elastic on foil
[{'uid': 'fc148109-7b0b-4193-91b8-e58345ada866',
  'fwhm': 2.9011180606603375,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -12300.0},
 {'uid': '704ca4e5-6d3e-481d-b951-e83353ccdcfa',
  'fwhm': 2.7307806616945527,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -9900.0},
 {'uid': '37ef7525-e02d-4b8c-a942-e89bb11ab899',
  'fwhm': 2.4222016766434535,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -7500.0},
 {'uid': 'fc42436b-2d71-46f9-ad1b-11d1c4938fb4',
  'fwhm': 2.271535088661949,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -5100.0},
 {'uid': '7865f00e-82d8-40b7-9092-e52a06d52532',
  'fwhm': 2.00544902767615,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -2700.0},
 {'uid': 'e48a7559-a076-49be-aa6b-2b2c4f3c25d5',
  'fwhm': 1.8584364404050575,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': -300.0},
 {'uid': '154391bf-5e78-4243-aede-8906227c4746',
  'fwhm': 1.783954573079427,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 2100.0},
 {'uid': '731f834a-a6f2-465b-9b50-43799596b928',
  'fwhm': 1.5156032852937642,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 4500.0},
 {'uid': '9bbbcd2b-4fa4-44a5-866a-938f140a7765',
  'fwhm': 1.4024573581864388,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 6900.0},
 {'uid': '9cf41dc8-b1a1-435c-8cef-60979e1a2cc8',
  'fwhm': 1.3853129494218592,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 9300.0},
 {'uid': 'ab905e84-5ca1-4166-b2ca-b05be03933f8',
  'fwhm': 1.3691874164233013,
  'tweak_motor_description': 'Johann Aux2 Crystal X',
  'tweak_motor_position': 11700.0}]


[{'uid': '9d15c670-a7ac-4b60-b7f4-c19dec7776b7',
  'fwhm': 79.5193314899443,
  'tweak_motor_description': 'Johann Aux4 Crystal X',
  'tweak_motor_position': -5000.0},
 {'uid': '3d285540-4333-4fa1-845a-6a4470496202',
  'fwhm': 79.21308886134102,
  'tweak_motor_description': 'Johann Aux4 Crystal X',
  'tweak_motor_position': -2500.0},
 {'uid': '5bfb4507-7a1c-494d-9564-fa727569a72d',
  'fwhm': 78.83613586202637,
  'tweak_motor_description': 'Johann Aux4 Crystal X',
  'tweak_motor_position': 0.0},
 {'uid': '64cb92f5-a13c-4ae5-903e-b2f362b9c13b',
  'fwhm': 79.87258246084957,
  'tweak_motor_description': 'Johann Aux4 Crystal X',
  'tweak_motor_position': 2500.0},
 {'uid': 'b8d48a67-b2a7-4836-9266-b9dc60b2596d',
  'fwhm': 82.78787885287375,
  'tweak_motor_description': 'Johann Aux4 Crystal X',
  'tweak_motor_position': 5000.0}]


x = xview_gui.widget_project
idxs = [i.row() for i in x.list_project.selectedIndexes()]
project = xview_gui.project
ref_idx = idxs[-1]
test_idxs = idxs[:-1]


energy_ref = project[ref_idx].energy
mu_ref = project[ref_idx].flat

e0 = 8979
# emin = e0 - 25
# emax = e0 + 25
emin = e0 - 5
emax = e0 + 5
energy_mask = (energy_ref >= emin) & (energy_ref <= emax)

chisq = []
roll = []
e_shift = []

for idx in test_idxs:
    energy_test = project[idx].energy
    mu_test = project[idx].flat

    _shift, _ = compute_shift_between_spectra(energy_test, mu_test, energy_ref[energy_mask], mu_ref[energy_mask])
    e_shift.append(_shift)
    mu_test_interp = np.interp(energy_ref, energy_test, mu_test)
    _chisq = np.sum((mu_test_interp - mu_ref)[energy_mask]**2)
    chisq.append(_chisq)

    _roll = float('.'.join(project[idx].name.split(' ')[3].split('_')))
    roll.append(_roll)

plt.figure(1, clear=True)
plt.plot(roll, chisq, 'ks')

plt.figure(2, clear=True)
plt.plot(roll, e_shift, 'ks')

########


for axis in ['x', 'y', 'roll', 'yaw']:
    k4 = f'motor_cr_aux4_{axis}'
    k5 = f'motor_cr_aux5_{axis}'
    rowland_circle.config['parking'][k4] = _config[k5]
    rowland_circle.config['parking'][k5] = _config[k4]







from numpy.polynomial import Polynomial


# uid = '53760d42-63d2-47ff-84c5-de111cbbb136'
# uid = 'e1d69459-8e4c-45b5-83b8-8306a2fa7107'
# uid = 'e6f6deef-848b-47ff-8414-a0755916e8f5'
# uid = 'fc2040af-af1a-465a-97ba-1567cc41b785'
# uid = '9b7ac035-d97c-460d-8d7b-ff541aa08738'
uid = '5e3919ef-13d2-4a97-8ebd-2bb870bd93fd'
# uid = '5066fde0-df7e-433e-995f-7bb256870675'
# uid = '39a6639c-cc2d-4df8-bb27-f16b6fafe431'
# uid = '928b333e-855d-48f2-95b1-c954cfc25919'

# uid = 'fc2040af-af1a-465a-97ba-1567cc41b785'
hdr = db[uid]

df = process_monitor_scan(db, uid, det_for_time_base='pil100k')

x_key, y_key = 'johann_aux3_crystal_motor_cr_aux3_roll', 'pil100k_stats1_total'
x = df[x_key].values
y = df[y_key].values
y = y - np.mean(y[:10])

y_max = np.mean(np.sort(y)[-5:])
mask = y >= y_max * 0.3

x_roi = x[mask]
y_roi = y[mask]

y_roi_log = np.log(y_roi)
#
# from sklearn.linear_model import BayesianRidge, RidgeCV
#
# x_roi_prep = (x_roi - np.mean(x_roi)) / np.std(x_roi)
# y_roi_prep = (y_roi_log - np.mean(y_roi_log)) / np.std(y_roi_log)
#
# n_order = 7
# X_roi = np.vander(x_roi_prep, n_order + 1, increasing=True)
# # reg = BayesianRidge(tol=1e-6, fit_intercept=False)
# # reg.set_params(alpha_init=1e3, lambda_init=1e-3)
#
# reg = RidgeCV(alphas=10**np.linspace(-5, 5, 25), fit_intercept=False, store_cv_values=True)
#
# reg.fit(X_roi, y_roi_prep)

# p = Polynomial.fit(x_roi_prep, y_roi_prep, n_order)

plt.figure(1, clear=True)
plt.plot(x_roi_prep, y_roi_prep)
plt.plot(x_roi_prep, reg.predict(X_roi))
plt.plot(x_roi_prep, p(x_roi_prep))

p = Polynomial.fit(x_roi, y_roi_log, 5)
# p = Polynomial(reg.coef_)
y_roi_log_fit = p(x_roi)
y_roi_fit = np.exp(y_roi_log_fit)

y_log_fit = p(x)
y_fit = np.exp(y_log_fit)

x_extrema = p.deriv().roots()
x_maxima = x_extrema[p.deriv().deriv()(x_extrema) < 0]
x_maxima = x_maxima[(x_maxima>=x_roi.min()) & (x_maxima<=x_roi.max())]

x_maximum = np.real(x_maxima[np.argmin(np.abs(x_maxima - x_roi[np.argmax(y_roi)]))])
y_maximum = np.exp(p(x_maximum))

def estimate_hm_positions_of_peak(x, y):
    x_cen = x[np.argmax(np.abs(y))]
    y_diff = np.abs(y - 0.5)
    x_low = x < x_cen
    x_high = x > x_cen
    x1 = np.interp(0.5, y[x_low], x[x_low])
    x2 = np.interp(0.5, y[x_high][::-1], x[x_high][::-1])
    return x1, x2

x1, x2 = estimate_hm_positions_of_peak(x_roi, y_roi_fit/y_maximum)

# _, fwhm = estimate_center_and_width_of_peak(x_roi, y_roi_fit)

plt.figure(1, clear=True)
plt.plot(x, y / y_maximum, 'k.-')
plt.plot(x_roi, y_roi / y_maximum, 'b.-')
plt.plot(x_roi, y_roi_fit / y_maximum, 'r-')

plt.vlines([x1, x2], 0, 1, colors='r')

# plt.plot(x_roi, p(x_roi), 'r-')

def _test_analyze_linewidth_fly_scan(db, uid, x_key='johann_main_crystal_motor_cr_main_roll', rois=None,
                                     plot_func=None):
    fname_bin = db[uid].start['interp_filename'][:-3] + 'dat'
    df, _ = load_binned_df_from_file(fname_bin)
    x = df[x_key].values
    if rois is None: rois = [1]

    for i in rois:
        field = f'pil100k_roi{i}'
        y = np.abs(df[field].values / df['i0'].values)

        y_max_act = np.mean(np.sort(y)[-5:])

        y = normalize_peak(y, bkg1=5, bkg2=-5, nmax=5)

        y_max = np.mean(np.sort(y)[-5:])
        mask = y >= y_max * 0.3

        x_roi = x[mask]
        y_roi = y[mask]

        com = np.sum(y_roi * x_roi) / np.sum(y_roi)
        variance = np.sum(y_roi * x_roi ** 2) / np.sum(y_roi) - com ** 2

    return variance, y_max_act


uids = range(390431+2, 390457+1, 2)
uids = range(390413, 390427+1, 2)
ALIGNMENT_DATA = []

for uid in uids:
    _johann_analyze_alignment_scan('epics_fly_scan_johann_emission_alignment_plan_bundle', _uid=uid, rois=[1],
                                   alignment_data=ALIGNMENT_DATA, fom=_scan_fom_dict['emission']['alignment']['fom'])
    x_key = db[uid].start['motor_stream_names'][0].replace('_monitor', '')
    variance, y_max = _test_analyze_linewidth_fly_scan(db, uid, x_key=x_key, rois=[1])
    ALIGNMENT_DATA[-1]['variance'] = variance
    ALIGNMENT_DATA[-1]['y_max'] = y_max

df = pd.DataFrame(ALIGNMENT_DATA)

plt.figure(1, clear=True)
get_optimal_crystal_alignment_position(df.tweak_motor_position, df.emission_line_fwhm, plot_func=my_plot_func)


plt.figure(2, clear=True)
get_optimal_crystal_alignment_position(df.tweak_motor_position, df.variance, plot_func=my_plot_func)


plt.figure(3, clear=True)
get_optimal_crystal_alignment_position(df.tweak_motor_position, -df.y_max, plot_func=my_plot_func)

hdr = db[389939]
t = hdr.table(stream_name='pil100k_stream', fill=True)
fig, ax = plt.subplots(1,100, figsize=(5,10))
for i, img in enumerate(t.pil100k_image[1]):
    ax[i].imshow(img)

elastic_scan = {}
for uid in range(389939, 389963):
    hdr = db[uid]
    x = hdr.start['tweak_motor_position']
    t = hdr.table(stream_name='pil100k_stream', fill=True)
    scan = []
    for img in t.pil100k_image[1]:
        scan.append(np.sum(img[85:101, 246:251]))
    elastic_scan[x] = scan

########
class Bla:
    '''class bla bla bla '''
    def __init__(self):
        '''
        bla bla bla for init
        '''
        self.value = 1



RE(fly_epics_scan_simple_johann_piezo_plan(crystal='main', axis='roll', scan_range=300, duration=4))
RE(fly_epics_scan_simple_johann_piezo_plan(crystal='main', axis='roll', scan_range=300, duration=10))

data=[
['main', 500.0,  9989.719, 2.303],
['aux2', 500.0,  9989.603, 2.279],
['aux3', 500.0,  9989.922, 2.353],
['aux4', 500.0,  9989.192, 2.374],
['aux5', 500.0,  9990.187, 2.437],
['main', 494.0,  9990.065, 2.824],
['aux2', 494.0,  9990.178, 3.349],
['aux3', 494.0,  9990.583, 3.409],
['aux4', 494.0,  9989.717, 2.736],
['aux5', 494.0,  9990.572, 2.512],
['main', 498.0,  9989.684, 2.321],
['aux2', 498.0,  9989.697, 2.567],
['aux3', 498.0,  9990.100, 2.521],
['aux4', 498.0,  9989.333, 2.335],
['aux5', 498.0,  9990.239, 2.315],
['main', 502.0,  9989.687, 2.347],
['aux2', 502.0,  9989.487, 2.295],
['aux3', 502.0,  9989.818, 2.297],
['aux4', 502.0,  9989.101, 2.483],
['aux5', 502.0,  9990.054, 2.818],
['main', 506.0,  9990.075, 2.975],
['aux2', 506.0,  9989.598, 2.642],
['aux3', 506.0,  9989.799, 2.582],
['aux4', 506.0,  9989.270, 3.314],
['aux5', 506.0,  9990.340, 4.193],
    ]

df = pd.DataFrame(data, columns=['crystal', 'R', 'cen', 'fwhm'])



reg_dict = \
{'bragg': {'motor_det_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_det_th1': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_det_th2': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_assy_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_assy_y': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_main_roll': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_main_yaw': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux2_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux2_y': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux2_roll': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux2_yaw': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux3_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux3_y': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux3_roll': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux3_yaw': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux4_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux4_y': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux4_roll': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux4_yaw': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux5_x': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux5_y': [73.06516837975127, 71.71225770356816, 73.06061662641574],
  'motor_cr_aux5_roll': [73.06516837975127,
   71.71225770356816,
   73.06061662641574],
  'motor_cr_aux5_yaw': [73.06516837975127,
   71.71225770356816,
   73.06061662641574]},
 'pos_nom': {'motor_det_x': [441.5864810814776,
   462.8381272263382,
   441.6590727125514],
  'motor_det_th1': [-7.021468191268235,
   -10.79841420807178,
   -7.034469319296711],
  'motor_det_th2': [-26.848195049229226,
   -25.777070384791898,
   -26.844297427871815],
  'motor_cr_assy_x': [946.5936121391642, 939.5241113423203, 946.5706396542035],
  'motor_cr_assy_y': [2.0, 2.0, 2.0],
  'motor_cr_main_roll': [3647.16837975127,
   2294.2577035681556,
   3642.6166264157364],
  'motor_cr_main_yaw': [-110.0, -110.0, -110.0],
  'motor_cr_aux2_x': [-3407.120729470286,
   -3493.148980670746,
   -3407.398231311373],
  'motor_cr_aux2_y': [-4808.887189205761,
   -4490.953809447376,
   -4807.834704958494],
  'motor_cr_aux2_roll': [3585.4275588244655,
   2253.4506183371527,
   3580.9445342933327],
  'motor_cr_aux2_yaw': [1708.8528994171945,
   2147.170068768113,
   1710.2370898281836],
  'motor_cr_aux3_x': [-3007.120729470286,
   -3093.148980670746,
   -3007.398231311373],
  'motor_cr_aux3_y': [-5908.887189205761,
   -5590.953809447376,
   -5907.834704958494],
  'motor_cr_aux3_roll': [4700.427558824465,
   3368.4506183371527,
   4695.944534293332],
  'motor_cr_aux3_yaw': [-2418.8528994171943,
   -2857.1700687681127,
   -2420.2370898281833],
  'motor_cr_aux4_x': [3979.59441552766,
   3638.5113583590473,
   3978.4955683127464],
  'motor_cr_aux4_y': [4377.113469528507, 5559.288265087447, 4381.024062290118],
  'motor_cr_aux4_roll': [5473.437049675518,
   4198.334085134112,
   5469.140577746956],
  'motor_cr_aux4_yaw': [3219.044078262905,
   4046.610626055618,
   3221.661770002467],
  'motor_cr_aux5_x': [4329.59441552766, 3988.5113583590473, 4328.495568312746],
  'motor_cr_aux5_y': [4027.1134695285073,
   5209.288265087447,
   4031.0240622901183],
  'motor_cr_aux5_roll': [3853.437049675518,
   2578.334085134112,
   3849.1405777469563],
  'motor_cr_aux5_yaw': [-2889.044078262905,
   -3716.610626055618,
   -2891.661770002467]},
 'pos_act': {'motor_det_x': [441.58642694, 462.8381015665, 441.659020681],
  'motor_det_th1': [-7.02180413783249,
   -10.798366637832487,
   -7.035085387832481],
  'motor_det_th2': [-26.848124999999996,
   -25.776562499999997,
   -26.844374999999996],
  'motor_cr_assy_x': [946.5936250555, 939.5240704275, 946.5706224145],
  'motor_cr_assy_y': [2.0000300000000006,
   2.0000300000000006,
   2.0000300000000006],
  'motor_cr_main_roll': [3094.911, 1720.0, 3068.928],
  'motor_cr_main_yaw': [-155.132, -115.22200000000001, -155.084],
  'motor_cr_aux2_x': [-3407.17, -3493.17, -3407.418],
  'motor_cr_aux2_y': [-4808.831, -4490.875, -4807.791],
  'motor_cr_aux2_roll': [2758.994, 1405.416, 2731.399],
  'motor_cr_aux2_yaw': [526.224, 704.392, 526.6510000000001],
  'motor_cr_aux3_x': [-3007.148, -3093.228, -3007.393],
  'motor_cr_aux3_y': [-5908.8060000000005, -5590.803, -5907.775000000001],
  'motor_cr_aux3_roll': [3668.367, 2316.764, 3637.083],
  'motor_cr_aux3_yaw': [-1550.958, -1849.483, -1551.925],
  'motor_cr_aux4_x': [3979.5460000000003, 3638.449, 3978.466],
  'motor_cr_aux4_y': [4377.21, 5559.539, 4381.149],
  'motor_cr_aux4_roll': [4980.1, 3704.523, 4955.281],
  'motor_cr_aux4_yaw': [1096.124, 1503.963, 1097.453],
  'motor_cr_aux5_x': [4329.568, 3988.4700000000003, 4328.463],
  'motor_cr_aux5_y': [4027.2690000000002, 5209.493, 4031.134],
  'motor_cr_aux5_roll': [2759.933, 1500.183, 2741.361],
  'motor_cr_aux5_yaw': [-1337.324, -1754.593, -1338.39]}}


motor = johann_main_crystal.motor_cr_main_roll
delta = 30
target = 2590
N = 100



motor = auxxy.x
delta = 0.1
target = 784
N = 100


motor.move(target)
ttime.sleep(0.25)
for i in range(N):
    print(f'{i} / {N}')
    motor.move(target - delta)
    ttime.sleep(0.05)
    motor.move(target)
    ttime.sleep(0.25)
    motor.move(target + delta)
    ttime.sleep(0.05)
    motor.move(target)
    ttime.sleep(0.25)


{'PI': 'Denis Leshchev', 'proposal': '313918', 'group': 'iss', 'scan_id': 104425, 'proposal_id': None, 'cycle': 2,
 'email': 'dleshchev@bnl.gov', 'Facility': 'NSLS-II', 'saf': '312165', 'slack_channel': 'C06MSP7H6CW', 'year': '2024', 'beamline_id': 'ISS (8-ID)', 'affiliation': 'NSLS II', 'experimenters': []}>




elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
            'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os'
             'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
             'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

edges = ['K','L1','L2','L3']

reference_dict = {}
for element in elements:
    edge_dict = {}

    for edge in edges:
        spectrum = []
        try:
            _foil_spectrum_tuple = db_proc.foil_spectrum(element, edge)
            _dataframe_foil = pd.DataFrame(np.column_stack(_foil_spectrum_tuple))
            _dataframe_foil = _dataframe_foil.sort_values(0)
            _nparray_foil = np.array(_dataframe_foil)
            spectrum = [ list(_nparray_foil[:,0]),list(_nparray_foil[:,1]) ]
        except:
            pass
        if spectrum != []:
            edge_dict[edge] = spectrum
    if edge_dict != {}:
        reference_dict[element]= edge_dict







#XIA data readout
fn = '/nsls2/data/iss/legacy/Sandbox/epics/test_001.h5'
import h5py
f= h5py.File(fn, 'r')
data = f['entry']['instrument']['detector']['data']




import numpy as np
import time
import sys
import os

# !/usr/bin/python
"""
Decodes the mapping buffers from the XIA xMXAP electronics
These can be MCA mapping mode, SCA mapping mode, or List mapping mode (not supported yet).
"""
import numpy as np
import time
import sys
import os


def aslong(d):
    """unravels and converts array of int16 (int) to int32 (long)"""
    # need to unravel the array!!!
    d = d.astype(np.int16).ravel()
    d.dtype = np.int32
    return d


class xMAPBufferHeader(object):
    def __init__(self, buff):
        self.tag0 = buff[0]  # Tag word 0
        self.tag1 = buff[1]  # Tag word 1
        self.headerSize = buff[2]  # Buffer header size
        #  Mapping mode (1=Full spectrum, 2=Multiple ROI, 3=List mode)
        self.mappingMode = buff[3]
        self.runNumber = buff[4]  # Run number
        # Sequential buffer number, low word first
        self.bufferNumber = aslong(buff[5:7])[0]
        self.bufferID = buff[7]  # 0=A, 1=B
        self.numPixels = buff[8]  # Number of pixels in buffer
        # Starting pixel number, low word first
        self.startingPixel = aslong(buff[9:11])[0]
        self.moduleNumber = buff[11]
        self.channelID = np.array(buff[12:20]).reshape((4, 2))
        self.channelSize = buff[20:24]
        self.bufferErrors = buff[24]
        self.userDefined = buff[32:64]
        self.listMode = buff[64]
        self.wordsPerEvent = buff[65]
        self.totalEvents = aslong(buff[66:68])[0]
        self.specialRecords = aslong(buff[68:70])[0]

    def report(self):
        print(["{}={}".format(field, getattr(self, field)) for field in ['mappingMode',
                                                                         'runNumber', 'bufferNumber', 'bufferID',
                                                                         'numPixels',
                                                                         'startingPixel', 'moduleNumber', 'channelSize',
                                                                         'totalEvents']])


class xMAPMCAPixelHeader(object):
    def __init__(self, buff):
        self.tag0 = buff[0]
        self.tag1 = buff[1]
        self.headerSize = buff[2]
        # Mapping mode (1=Full spectrum, 2=Multiple ROI, 3=List mode)
        self.mappingMode = buff[3]
        self.pixelNumber = aslong(buff[4:6])[0]
        self.blockSize = aslong(buff[6:8])[0]
        self.channelSize = buff[10:13]


class xMAPData(object):
    def __init__(self, npix, nmod, nchan):
        ndet = 4 * nmod
        self.firstPixel = 0
        self.numPixels = 0
        self.counts = np.zeros((npix, ndet, nchan), dtype='i2')
        self.realTime = np.zeros((npix, ndet), dtype='i8')
        self.liveTime = np.zeros((npix, ndet), dtype='i8')
        self.inputCounts = np.zeros((npix, ndet), dtype='i4')
        self.outputCounts = np.zeros((npix, ndet), dtype='i4')


CLOCKTICK = 0.320  # xmap clocktick = 320 ns


def decode_xmap_buffers(array_data):
    # array_data will normally be 3d:
    #  shape = (narrays, nmodules, buffersize)
    # but nmodules and narrays could be 1, so that
    # array_data could be 1d or 2d.
    #
    # here we force the data to be 3d
    shape = array_data.shape
    if len(shape) == 1:
        array_data.shape = (1, 1, shape[0])
    elif len(shape) == 2:
        array_data.shape = (1, shape[0], shape[1])

    narrays, nmodules, buffersize = array_data.shape
    modpixs = int(max(124, array_data[0, 0, 8]))
    npix_total = 0
    # real / live times are returned in microseconds.
    for array in range(narrays):
        for module in range(nmodules):
            d = array_data[array, module, :]
            bh = xMAPBufferHeader(d)
            dat = d[256:].reshape(modpixs, int((d.size - 256) / modpixs))
            # converting buffer from flat to

            npix = bh.numPixels
            if module == 0:
                npix_total += npix
                if array == 0:
                    # first time through, (array,module)=(0,0) we
                    # read mapping mode, set up how to slice the
                    # data, and build data arrays in xmapdat
                    mapmode = dat[0, 3]
                    if mapmode == 1:  # mapping, full spectra
                        nchans = d[20]
                        data_slice = slice(256, 8448)
                    elif mapmode == 2:  # ROI mode
                        # Note:  nchans = number of ROIS !!
                        nchans = max(d[264:268])
                        data_slice = slice(64, 64 + 8 * nchans)
                    xmapdat = xMAPData(narrays * modpixs, nmodules, nchans)
                    print(narrays * modpixs, nmodules, nchans)
                    xmapdat.firstPixel = bh.startingPixel

            # acquisition times and i/o counts data are stored
            # as longs in locations 32:64
            t_times = aslong(dat[:npix, 32:64]).reshape(npix, 4, 4)
            mod_slice = slice(module * 4, module * 4 + 4)
            p1 = npix_total - npix
            p2 = npix_total
            xmapdat.realTime[p1:p2, mod_slice] = t_times[:, :, 0]
            xmapdat.liveTime[p1:p2, mod_slice] = t_times[:, :, 1]
            xmapdat.inputCounts[p1:p2, mod_slice] = t_times[:, :, 2]
            xmapdat.outputCounts[p1:p2, mod_slice] = t_times[:, :, 3]

            # the data, extracted as per data_slice and mapmode
            t_data = dat[:npix, data_slice]
            if mapmode == 2:
                t_data = aslong(t_data)
            xmapdat.counts[p1:p2, mod_slice, :] = t_data.reshape(npix, 4, nchans)

    xmapdat.numPixels = npix_total
    xmapdat.counts = xmapdat.counts[:npix_total]
    xmapdat.realTime = CLOCKTICK * xmapdat.realTime[:npix_total]
    xmapdat.liveTime = CLOCKTICK * xmapdat.liveTime[:npix_total]
    xmapdat.inputCounts = xmapdat.inputCounts[:npix_total]
    xmapdat.outputCounts = xmapdat.outputCounts[:npix_total]
    return xmapdat






