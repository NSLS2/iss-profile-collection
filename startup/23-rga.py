print(ttime.ctime() + ' >>>> ' + __file__)

from ophyd import Device, Component as Cpt, EpicsSignal, EpicsSignalRO


class HidenRGA(Device):
    """Hiden RGA ophyd device. Reads partial-pressure (MID) values
    from the caproto IOC and records mass assignments as config."""

    # MID partial-pressure readback PVs (main data signals)
    mid1  = Cpt(EpicsSignalRO, 'P:MID1-I',  kind='normal', labels={'RGA'})
    mid2  = Cpt(EpicsSignalRO, 'P:MID2-I',  kind='normal', labels={'RGA'})
    mid3  = Cpt(EpicsSignalRO, 'P:MID3-I',  kind='normal', labels={'RGA'})
    mid4  = Cpt(EpicsSignalRO, 'P:MID4-I',  kind='normal', labels={'RGA'})
    mid5  = Cpt(EpicsSignalRO, 'P:MID5-I',  kind='normal', labels={'RGA'})
    mid6  = Cpt(EpicsSignalRO, 'P:MID6-I',  kind='normal', labels={'RGA'})
    mid7  = Cpt(EpicsSignalRO, 'P:MID7-I',  kind='normal', labels={'RGA'})
    mid8  = Cpt(EpicsSignalRO, 'P:MID8-I',  kind='normal', labels={'RGA'})
    mid9  = Cpt(EpicsSignalRO, 'P:MID9-I',  kind='normal', labels={'RGA'})
    mid10 = Cpt(EpicsSignalRO, 'P:MID10-I', kind='normal', labels={'RGA'})

    # Mass config PVs (different prefix, use add_prefix=() for absolute PV names)
    mass1  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID1',  add_prefix=(), kind='config')
    mass2  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID2',  add_prefix=(), kind='config')
    mass3  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID3',  add_prefix=(), kind='config')
    mass4  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID4',  add_prefix=(), kind='config')
    mass5  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID5',  add_prefix=(), kind='config')
    mass6  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID6',  add_prefix=(), kind='config')
    mass7  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID7',  add_prefix=(), kind='config')
    mass8  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID8',  add_prefix=(), kind='config')
    mass9  = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID9',  add_prefix=(), kind='config')
    mass10 = Cpt(EpicsSignalRO, 'XF:08IDB-VA{RGA:1}Mass:MID10', add_prefix=(), kind='config')

    # Control PVs (omitted from data stream; for programmatic use)
    open_exp  = Cpt(EpicsSignal, ':OpenExp',  kind='omitted')
    exp_name  = Cpt(EpicsSignal, ':ExpName',  kind='omitted', string=True)
    acquire   = Cpt(EpicsSignal, ':Acquire',  kind='omitted')
    run_exp   = Cpt(EpicsSignal, ':RunExp',   kind='omitted')
    abort_exp = Cpt(EpicsSignal, ':AbortExp', kind='omitted')
    close_exp = Cpt(EpicsSignal, ':CloseExp', kind='omitted')

    def read_config_metadata(self):
        """Return mass-to-channel mapping for scan metadata.
        Called by get_detector_md() in startup/60-plan_metadata.py."""
        md = {'device_name': self.name}
        for i in range(1, 11):
            md[f'mass{i}'] = getattr(self, f'mass{i}').get()
        return md


try:
    rga = HidenRGA('XF:08IDB-SE{RGA:1}', name='rga')
    rga.wait_for_connection(timeout=10)
except Exception as e:
    print(f'({ttime.ctime()}) RGA not available: {e}')
    rga = None
