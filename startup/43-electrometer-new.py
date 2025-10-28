class Electrometer(Device):

    polarity = 'neg'

    ch1 = Cpt(EpicsSignal, 'Current1:MeanValue_RBV')
    ch2 = Cpt(EpicsSignal, 'Current2:MeanValue_RBV')
    ch3 = Cpt(EpicsSignal, 'Current3:MeanValue_RBV')
    ch4 = Cpt(EpicsSignal, 'Current4:MeanValue_RBV')
    # ch5 = Cpt(EpicsSignal, 'Current5')
    # ch6 = Cpt(EpicsSignal, 'Current6')
    # ch7 = Cpt(EpicsSignal, 'Current7')
    # ch8 = Cpt(EpicsSignal, 'Current8')

    # ch1_range = Cpt(EpicsSignal, 'ADC:Range:A-SP')
    # ch2_range = Cpt(EpicsSignal, 'ADC:Range:D-SP')
    # ch3_range = Cpt(EpicsSignal, 'ADC:Range:C-SP')
    # ch4_range = Cpt(EpicsSignal, 'ADC:Range:D-SP')
    # ch5_range = Cpt(EpicsSignal, 'ADC:Range:E-SP')
    # ch6_range = Cpt(EpicsSignal, 'ADC:Range:F-SP')
    # ch7_range = Cpt(EpicsSignal, 'ADC:Range:G-SP')
    # ch8_range = Cpt(EpicsSignal, 'ADC:Range:H-SP')




    # ch1_offset = Cpt(EpicsSignal, 'Ch1:User:Offset-SP', kind=Kind.config)
    # ch2_offset = Cpt(EpicsSignal, 'Ch2:User:Offset-SP', kind=Kind.config)
    # ch3_offset = Cpt(EpicsSignal, 'Ch3:User:Offset-SP', kind=Kind.config)
    # ch4_offset = Cpt(EpicsSignal, 'Ch4:User:Offset-SP', kind=Kind.config)
    # ch5_offset = Cpt(EpicsSignal, 'Ch5:User:Offset-SP', kind=Kind.config)
    # ch6_offset = Cpt(EpicsSignal, 'Ch6:User:Offset-SP', kind=Kind.config)
    # ch7_offset = Cpt(EpicsSignal, 'Ch7:User:Offset-SP', kind=Kind.config)
    # ch8_offset = Cpt(EpicsSignal, 'Ch8:User:Offset-SP', kind=Kind.config)

    acquire = Cpt(EpicsSignal, 'Acquire', kind=Kind.omitted)
    acquire_mode = Cpt(EpicsSignal, 'AcquireMode', kind=Kind.omitted)
    # acquiring = Cpt(EpicsSignal, 'FA:Busy-I', kind=Kind.omitted)


    # divide = Cpt(EpicsSignal, 'FA:Divide-SP')
    # sample_len = Cpt(EpicsSignal, 'FA:Samples-SP')
    # wf_len = Cpt(EpicsSignal, 'FA:Wfm:Length-SP')
    #
    # stream = Cpt(EpicsSignal, 'FA:Stream-SP', kind=Kind.omitted)
    # streaming = Cpt(EpicsSignal, 'FA:Streaming-I', kind=Kind.omitted)
    #
    # acq_rate= Cpt(EpicsSignal,'FA:Rate-I', kind=Kind.omitted)
    # stream_samples = Cpt(EpicsSignal, 'FA:Stream:Samples-SP')
    #
    # filename_bin = Cpt(EpicsSignal, 'FA:Stream:Bin:File-SP')
    # filebin_status = Cpt(EpicsSignal, 'FA:Stream:Bin:File:Status-I')
    #
    # trig_source = Cpt(EpicsSignal, 'Machine:Clk-SP')


em2 = Electrometer('XF:08ID1-ES{EM:01}', name = 'em2')
