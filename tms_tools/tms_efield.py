import matplotlib.pyplot as plt
import numpy as np
from math import pi, floor

class shape():
    def __init__(self, 
        shape: str = "Sine", 
        width: float = 100e-6, 
    ):
        #from sympy import Symbol, Function, diff, lambdify, sin
        import sympy as sp

        # Shape can be "Sine", "Damped_Sine", "Monophasic", "Square"
        # Only "Sine" and "Square" are supported currently
        self.shape = shape
        self.width = width
        if self.shape not in ["Sine", "Square"]:
            raise ValueError("The only shapes supported currently are 'Sine' and 'Square'")
        if self.width <= 0:
            raise ValueError("Width must be > 0")
        t = sp.Symbol('t')
        f = sp.Function(t)
        if self.shape == "Sine":
            scale = 2*pi/width
            f = -sp.sin(t*scale)/scale
        elif self.shape == "Square":
            f = t
        self.evalf = sp.lambdify(t, f)
        self.evaldfdt = sp.lambdify(t, sp.diff(f, t))

class pattern():
    def __init__(self, 
        pshape: shape,                      # Shape of the pulse
        pattern: str = "Single",            # Pattern of the set
        count: int = None,                  # Number of pulses in a set
        interval: float = None,             # Duration of interval between pulses in a set
        p_onset_interval: float = None,     # Duration of interval between onset of pulses in a set
        set_freq: float = None,             # Frequency of pulse onsets in a set
        rd: int = 9                         # Rounding precision (up to 10^-rd seconds)
    ):
        # Pattern can be "Single" if count = 1 (then interval, p_onset_interval, & set_freq are meaningless)
                       # "TBS" if count = 3 (Theta burst stimulation)
                       # "Unique" if count is any other value
        # interval = p_onset_interval - pshape.width
        # p_onset_interval = 1/set_freq
        # Highest priority when defined | interval > p_onset_interval > set_freq | lowest priority

        self.pshape = pshape
        width = self.pshape.width
        self.pattern = pattern
        if self.pattern == "Single":
            interval = 0
            count = 1
        if self.pattern == "TBS":
            count = 3
        if interval == None:
            if p_onset_interval == None:
                if set_freq == None:
                    raise ValueError("Pulse interval, onset interval, or set frequency of pattern must be defined")
                else:
                    if set_freq > 1/width: raise ValueError(f"Set freq {set_freq} Hz must be <= 1/(pulse width) {1/width} Hz")
                    interval = 1/set_freq - width
            else:
                if p_onset_interval < width: raise ValueError(f"Pulse onset interval {p_onset_interval} must be >= pulse width {width}")
                interval = p_onset_interval - width
        if interval < 0: raise ValueError(f"Pulse interval of pattern {interval} must be >= 0")
        self.rd = rd
        self.interval = round(interval, rd)
        self.p_onset_interval = round(self.interval + width, rd)
        self.set_freq = round(1/self.p_onset_interval, rd)
        #print(f"Interval: {self.interval} s, width: {width} s, onset int: {self.p_onset_interval} s, set freq {self.set_freq} Hz")
        self.count = count
        if self.pattern not in ["Single", "TBS", "Unique"]:
            raise ValueError("Pulse pattern must be categorized as either 'Single', 'TBS', or 'Unique'")
        if self.count == None: raise ValueError("Pulse count of pattern must be defined")
        elif self.count < 1: raise ValueError("Pulse count of pattern must be >= 1")

def efield(
        freq: float,        # Frequency of pulse sets
        duration: float,    # Duration of waveform
        dt: float,          # Duration of time step
        tstart: float,      # Initial waiting period
        ef_amp: float,      # Amplitude of the max E-field in the desired waveform
        pat: pattern        # Pattern object containing data on the waveform
):
    #                                                             All time units in seconds, frequency units in Hz
    rd = pat.rd                                                 # Rounding precision (up to 10^-rd seconds)
    pulse = pat.pshape.evaldfdt                                 # E-field pulse function (SymPy)
    pwidth = pat.pshape.width                                   # Duration of one pulse
    p_onset_interval = pat.p_onset_interval                     # Duration of interval between the onset of pulses in a set
    inter_p_interval = pat.interval                             # Duration of interval between pulses within a set
    count = pat.count                                           # Number of pulses in one set
    set_width = (count-1)*inter_p_interval + count*pwidth       # Duration of one set of pulses
    set_onset_interval = 1/freq                                 # Duration of interval between the onset of sets of pulses
    inter_set_interval = set_onset_interval - set_width         # Duration of interval between sets of pulses

    # Check that inputs are valid
    if duration <= 0:
        raise ValueError("Duration must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")    
    if inter_set_interval < 0:
        raise ValueError(f"Duration of pulse set [{set_width} s] must be <= interval between pulse onset (1/frequency) [{set_onset_interval} s]")
    
    # Initialize variables for waveform construction
    wav = []                        # E-field waveform at each point in time

    duration = round(duration, rd)
    dt = round(dt, rd)

    # Construct list of time points
    nstep = int(round(duration/dt))     # Number of time steps within the duration
    npoint = nstep+1                    # Number of time points within the duration
    time = list(np.linspace(0, duration, npoint))
    time = [round(time[i], rd) for i in range(npoint)]
    tind = 0                            # Index of current time point in time course

    if tstart > duration:
        # Waveform is silent for entire duration
        wav = list(np.zeros(npoint))
        return [wav, time]

    # Write waiting period to waveform if necessary
    nstepstart = int(round(tstart/dt))  # Number of time points until tstart
    while tind < nstepstart:
        wav.append(0)
        tind += 1

    # Write waveform
    tpulse_start = tstart           # Time of current pulse start
    while tind <= nstep:
        # Determine whether currently within an active pulse
        twav = round(time[tind] - tstart, rd)           # Time passed since tstart
        tset = round(twav % set_onset_interval, rd)     # Time passed since most recent set onset
        cset = int(floor(twav/set_onset_interval))      # Number of sets completed
        tpulse = round(tset % p_onset_interval, rd)     # Time passed since most recent pulse onset
        cpulse = int(floor(tset/p_onset_interval))      # Number of pulses completed (including inter-pulse-interval) within set
        
        if tset <= set_width and tpulse <= pwidth:
            # Within a pulse
            tpulse_start = tstart + cset*set_onset_interval + cpulse*p_onset_interval
            wav.append(pulse(time[tind]-tpulse_start)*ef_amp)
        else:
            wav.append(0)

        tind += 1
    return [wav, time]

def get_efield(
        freq: float,
        duration: float,
        dt: float = 25e-3,
        tstart: float = 0.0,
        ef_amp: float = 100.0,
        pshape: str = "Sine", 
        width: float = 100e-3, 
        pat: str = "Single",
        count: int = None,
        interval: float = None,
        p_onset_interval: float = None,
        set_freq: float = None,
        plot: bool = False,
):
    '''
    Takes all time units specified in ms
    Takes all frequency units specified in Hz
    Returns waveform in mV/um (or V/mm)
    Returns time course in ms
    '''

    if isinstance(interval, float):
        interval /= 1e3                     # Convert from ms to s
    if isinstance(p_onset_interval, float):
        p_onset_interval /= 1e3             # Convert from ms to s

    wav, time = efield(freq=freq, 
                  duration=duration/1e3,    # Convert from ms to s
                  dt=dt/1e3,                # Convert from ms to s
                  tstart=tstart/1e3,        # Convert from ms to s
                  ef_amp=ef_amp, 
                  pat=pattern(
                    pshape=shape(
                      shape=pshape, 
                      width=width/1e3),     # Convert from ms to s
                    pattern=pat, 
                    count=count,
                    interval=interval,
                    p_onset_interval=p_onset_interval,
                    set_freq=set_freq))
    
    wav = [v*1e-3 for v in wav]             # Convert from V/m to mV/um (or V/mm)
    time = [t*1e3 for t in time]            # Convert from s to ms

    if plot:
        plt.figure()
        plt.step(time, wav, where='post')
        plt.xlabel('Time (ms)')
        plt.ylabel('E-field (V/m)')
        plt.show()

    return wav, time

# Testing
def main():
    get_efield(freq=10, 
            duration=500e-3,
            width=100e-3,
            pshape="Sine",
            pat="Single",
            #count=2,
            #interval=100e-6,
            )

    plt.show()

    ''',
    dt=25e-6,
    pat="Unique",
    count=2,
    interval=100e-6,
    p_onset_interval=None,
    set_freq=None'''

    """plt.step(time, wav, where='post')
    plt.show()"""

if __name__ == '__main__':
    main()