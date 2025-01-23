import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi, ceil, floor
from .units import *
from warnings import warn, simplefilter

class Shape:
    def __init__(
            self,
            shape: str,
            pulse_width_ms: float,
        ):
        # from sympy import Symbol, Function, diff, lambdify, sin
        import sympy as sp

        # Shape can be "Ideal_Sine", "Ideal_Square", "Biphasic", "Half-Biphasic", "Monophasic"
        # Only "Ideal_Sine" and "Ideal_Square" are supported currently
        self.shape = shape
        self.pulse_width_ms = pulse_width_ms

        # Define efield waveform shape normalized to a value of 1; scaling comes later        
        if self.shape == "Ideal_Sine":
            sin_freq_kHz = 2 * pi / pulse_width_ms # Angular frequency of sine wave
            self.efield_waveform = lambda t: math.cos(t * sin_freq_kHz) # Sinusoidal TMS pulses have cosinusoidal E-field waveforms
        elif self.shape == "Ideal_Square":
            self.efield_waveform = lambda t: 1 # Constant value of 1
        elif self.shape == "Biphasic":
            self.efield_waveform = coil_recording('b', self.pulse_width_ms)
        elif self.shape == "Half-Sine":
            self.efield_waveform = coil_recording('h', self.pulse_width_ms)
        elif self.shape == "Monophasic":
            self.efield_waveform = coil_recording('m', self.pulse_width_ms)
        else:
            raise ValueError(f"Pulse shape [{self.shape}] must be \"Ideal_Sine\" or \"Ideal_Square\"")

def coil_recording(shape_char, pulse_width_ms):
    '''
    Returns a lambda function for the E-field waveform based on real data from TMS coils
    Recordings from a MagPro X100 stimulator with a MagVenture MCF-B70 figure-of-eight coil sampled at 5 MHz
    Data copied from TMS-Neuro-Sim (Weise et. al. 2023)

    Currently effectively cuts off the waveform past width_rec, which means leaving some stimulation artifacts
    unrepresented in the simulation. Particularly for monophasic pulses, a pulse width is poorly defined and the
    cutoff point leaves out a period of low-amplitude negative field. Future implementations might accept a definition of
    desired pulse width while still modeling the remainder of the pulse after the defined width is over.
    '''
    from scipy.io import loadmat
    import os

    curr_dir = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    mat = loadmat('TMSwaves.mat')
    os.chdir(curr_dir)
    Erec = [val[0] for val in mat[f'Erec_{shape_char}']] # Recorded E-field waveform; Convert matlab data structure into list
    trec = [round(val[0], 9) for val in mat[f't{shape_char}']] # Time points recorded; Also correcting floating point error
    width_rec = 0.3 if shape_char in ['b', 'm'] else 0.15
    width_scalar = pulse_width_ms/width_rec
    trec = [t*width_scalar for t in trec]
    return lambda t: np.interp(t, trec, Erec)

class Pattern:
    def __init__(
            self,
            pulse_shape: Shape,  # Shape of the pulse
            num_pulses_per_burst: int,  # Number of pulses in a burst
            pulse_interval_within_burst_ms: float | None = None,  # Duration of interval between pulses in a burst
            pulse_onset_interval_within_burst_ms: float | None = None,  # Duration of interval between onset of pulses in a burst
            pulse_freq_within_burst_Hz: float | None = None,  # Frequency of pulse onsets in a burst
        ):
        """If pattern is "Single" then num_pulses_per_burst set to 1 (and pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, & pulse_freq_within_burst_Hz are meaningless)
        If "TBS" then num_pulses_per_burst set to 3 (Theta burst stimulation)

        pulse_interval_within_burst_ms = pulse_onset_interval_within_burst_ms - pulse_shape.pulse_width_ms
        pulse_onset_interval_within_burst_ms = 1/pulse_freq_within_burst_Hz
        Highest priority when defined | pulse_interval_within_burst_ms > pulse_onset_interval_within_burst_ms > pulse_freq_within_burst_Hz | lowest priority"""

        # Set attributes
        self.pulse_shape = pulse_shape
        pulse_width_ms = self.pulse_shape.pulse_width_ms
        
        # Check that num_pulses_per_burst and pattern are valid
        if type(num_pulses_per_burst) != int or num_pulses_per_burst < 1:
            raise ValueError(
                f"num_pulses_per_burst [{num_pulses_per_burst}] must be defined as a positive non-zero integer"
                )
        self.num_pulses_per_burst = num_pulses_per_burst

        if num_pulses_per_burst == 1:
            pulse_interval_within_burst_ms = 0 # Meaningless, but marks it as set

        # Check that pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, and pulse_freq_within_burst_Hz are valid
        if pulse_interval_within_burst_ms == None:
            if pulse_onset_interval_within_burst_ms == None:
                if pulse_freq_within_burst_Hz == None:
                    raise ValueError(
                        "pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, or pulse_freq_within_burst_Hz" \
                            f"must be defined if num_pulses_per_burst [{num_pulses_per_burst}] > 1"
                    )
                else:
                    if pulse_freq_within_burst_Hz > 1/pulse_width_ms * kHz: # Comparison in Hz
                        raise ValueError(
                            f"pulse_freq_within_burst_Hz [{pulse_freq_within_burst_Hz}] must be <= 1/(pulse_width_ms) [{1/pulse_width_ms * kHz}]"
                        )
                    pulse_interval_within_burst_ms = 1/pulse_freq_within_burst_Hz * s - pulse_width_ms
            else:
                if pulse_onset_interval_within_burst_ms < pulse_width_ms:
                    raise ValueError(
                        f"pulse_onset_interval_within_burst_ms [{pulse_onset_interval_within_burst_ms}] must be >= pulse_width_ms [{pulse_width_ms}]"
                    )
                pulse_interval_within_burst_ms = pulse_onset_interval_within_burst_ms - pulse_width_ms
        if pulse_interval_within_burst_ms < 0:
            raise ValueError(f"pulse_interval_within_burst_ms [{pulse_interval_within_burst_ms}] must be >= 0")
        
        # Set remaining attributes
        self.pulse_interval_within_burst_ms = pulse_interval_within_burst_ms
        self.pulse_onset_interval_within_burst_ms = self.pulse_interval_within_burst_ms + pulse_width_ms
        self.pulse_freq_within_burst_Hz = 1 / self.pulse_onset_interval_within_burst_ms * kHz


def generate_efield(
        burst_freq_Hz: float | None,  # Frequency of pulse bursts in Hz (meaningless for sTMS & tDCS? (TODO), in which case this is None)
        simulation_duration_ms: float,  # Duration of waveform
        default_dt: float,  # Duration of time step in ms
        stim_start_ms: float,  # Initial waiting period
        stim_end_ms: float | None, # Time when stimulation ends
        total_num_tms_pulse_bursts: int | None, # Number of pulse bursts after stim_start_ms
        efield_amplitude_mV_per_um: float,  # Amplitude of the max E-field in the desired waveform
        pat: Pattern,  # Pattern object containing data on the waveform
        pulse_dt: float, # Duration of time step in ms when electric field activity is present; defaults to be equivalent to dt
        buffer_size_ms: float = 1e-3, # Minimum time buffer between silent and active periods (improves accuracy of interpolation)
        rd: int = 9, # Rounding precision to correct floating point error (rounds to 10^-rd ms)
    ):
    efield_waveform = pat.pulse_shape.efield_waveform  # E-field pulse function (SymPy)
    pulse_width_ms = pat.pulse_shape.pulse_width_ms  # Duration of one pulse
    pulse_onset_interval_within_burst_ms = pat.pulse_onset_interval_within_burst_ms # Duration of interval between the onset of pulses in a burst
    inter_p_interval_ms = pat.pulse_interval_within_burst_ms # Duration of interval between pulses within a burst
    num_pulses_per_burst = pat.num_pulses_per_burst # Number of pulses in one burst
    burst_width_ms = round((num_pulses_per_burst - 1) * inter_p_interval_ms + num_pulses_per_burst * pulse_width_ms, rd) # Duration of one burst of pulses (time of intervals + time of pulses)

    # Check parameters
    if total_num_tms_pulse_bursts != None:
        if total_num_tms_pulse_bursts <= 1:
            burst_freq_Hz = None
        elif burst_freq_Hz == None:
            raise ValueError(f"rtms_pulse_burst_freq_Hz must be defined if total_num_tms_pulse_bursts [{total_num_tms_pulse_bursts}] > 1")
            # Situation only applicable/possible with rTMS

    if burst_freq_Hz != None:
        burst_onset_interval_ms = round(1 / burst_freq_Hz * s, rd) # Duration of interval between the onset of bursts of pulses
        inter_burst_interval_ms = round(burst_onset_interval_ms - burst_width_ms, rd) # Duration of interval between bursts of pulses
    else:
        burst_onset_interval_ms = None
        inter_burst_interval_ms = None

    if inter_burst_interval_ms != None:
        if inter_burst_interval_ms < 0:
            raise ValueError(
                f"Duration of pulse burst [{burst_width_ms} ms] must be <= interval between pulse burst onset " \
                    f"(1/rtms_pulse_burst_freq_Hz or 1/tacs_freq_Hz) [{burst_onset_interval_ms} ms]"
            )

    # Construct waveform of electric field, taking advantage of linear interpolation between time points

    # Initialize variables for waveform construction
    time = [] # Points in time
    wav = []  # E-field waveform at each point in time

    # Build burst_start_times_ms
    cur_t = stim_start_ms # Current time as progressing through loop
    burst_start_times_ms = [] # List of burst start times
    num_bursts = 0 # Number of bursts accounted for

    while cur_t < stim_end_ms: # Iterate through stim duration and build burst_start_times_ms
        burst_start_times_ms.append(cur_t)
        num_bursts += 1
        if total_num_tms_pulse_bursts != None: # If using total_num_tms_pulse_bursts
            if num_bursts >= total_num_tms_pulse_bursts: # And the number of bursts has reached the total limit
                break # End the building process
        cur_t += burst_onset_interval_ms # If not, advance to the start of the next burst

    pulse_start_times_ms = [] # List of pulse start times
    for burst_start_time in burst_start_times_ms:
        for pulsenum in range(num_pulses_per_burst):
            pulse_start_times_ms.extend([burst_start_time + pulse_onset_interval_within_burst_ms*pulsenum])

    pulse_end_times_ms = [round(start_time + pulse_width_ms, rd) for start_time in pulse_start_times_ms]
    for ind, pulse_end_time in reversed(list(enumerate(pulse_end_times_ms))): # Reversed so that pop(ind) does not cause index shift
        if pulse_end_time in pulse_start_times_ms:
            pulse_end_times_ms.pop(ind)

    # Define a single pulse (noninclusive to last time step)
    sampled_pulse_time_steps = ceil(pulse_width_ms / pulse_dt) - 1
    sampled_pulse_width_ms = sampled_pulse_time_steps * pulse_dt # Effective length of pulse when accounting for sampling resolution & excluding last time step
    npoints_pulse = sampled_pulse_time_steps + 1  # Number of time points within a pulse
    if pat.pulse_shape.shape == "Ideal_Square":
        npoints_pulse = 2 # Only need the first and last time point of the pulse, as it does not change over the duration
    
    pulse_t = np.linspace(0, sampled_pulse_width_ms, npoints_pulse) # Time points within a pulse starting at t=0
    pulse = [efield_waveform(t) * efield_amplitude_mV_per_um for t in pulse_t] # Sample points of the pulse waveform; scale by efield_amplitude_mV_per_um
    
    if stim_start_ms > 0:
        # Start of initial silent period
        time.append(0)
        wav.append(0)

    pulse_end_time = 0
    for i, pulse_start in enumerate(pulse_start_times_ms):
        # Pre-pulse buffer
        last_silent_t = pulse_start - buffer_size_ms # Last time point of silent period before pulse start
        if last_silent_t > pulse_end_time: # If we've advanced far enough to need to specify the end of the silent period
            # Write end of silent period
            time.append(last_silent_t)
            wav.append(0)

        # Write pulse
        time.extend(pulse_t + pulse_start)
        wav.extend(pulse)

        # Post-pulse buffer
        pulse_end_time = pulse_start + pulse_width_ms # Time of the end of the pulse
        if (i == len(pulse_start_times_ms)-1 # If this is the last pulse
            or pulse_end_time < pulse_start_times_ms[i+1] - buffer_size_ms): # or if we will advance far enough to need to specify the start of a silent period
            # Write start of silent period
            time.append(pulse_end_time)
            wav.append(0)
    # Buffers necessary for simulation time steps when outside of a pulse

    time = [round(t, rd) for t in time] # Correct floating point error

    if time[-1] < simulation_duration_ms: # If the time course does not last the full duration
        # Place a silent period until the end of the simulation
        time.append(simulation_duration_ms)
        wav.append(0)

    if time[-1] > simulation_duration_ms: # If the time course is longer than the full duration
        # Trim the time course to fit the duration to save resources
        # Find index of the last time point less than the duration
        ind_last_t = 0
        for ind, t in reversed(list(enumerate(time))): # Going backwards will probably take less computation
            if t < simulation_duration_ms:
                ind_last_t = ind
                break
        time = time[:ind_last_t+2] # +2 because we want to still include the last point before and the first point after the duration
        wav = wav[:ind_last_t+2]

    def NetPyNE_interval_func(t):
        from neuron import h
        t = round(t, 9)
        
        prev_pulse_starts = [start for start in pulse_start_times_ms if start <= t]
        ind_last_pulse_start = len(prev_pulse_starts)-1

        if ind_last_pulse_start == -1: # If there have not been any pulses yet
            last_pulse_start = -1
        else:
            last_pulse_start = pulse_start_times_ms[ind_last_pulse_start]
        
        if ind_last_pulse_start + 1 < len(pulse_start_times_ms): # If there are more pulse start events
            next_pulse_start = pulse_start_times_ms[ind_last_pulse_start+1] # Set next_pulse_start
        else:
            next_pulse_start = -1 # Else flag next as invalid

        # Same for pulse end events
        prev_pulse_ends = [end for end in pulse_end_times_ms if end <= t]
        ind_last_pulse_end = len(prev_pulse_ends)-1

        if ind_last_pulse_end == -1: # If there have not been any pulses yet
            last_pulse_end = 0
        else:
            last_pulse_end = pulse_end_times_ms[ind_last_pulse_end]

        if ind_last_pulse_end + 1 < len(pulse_end_times_ms):
            next_pulse_end = pulse_end_times_ms[ind_last_pulse_end+1]
        else:
            next_pulse_end = -1

        # Set h.dt
        within_pulse = last_pulse_start >= last_pulse_end # Within a pulse if the last time a pulse started was more recent than the last time one ended
        if within_pulse:
            h.dt = pulse_dt
        else:
            h.dt = default_dt
        
        if next_pulse_start==-1:        # If there are no more pulse starts
            next_event=next_pulse_end   # Next event must be a pulse end or there are no more events (next_event=-1)
        elif next_pulse_end==-1:        # If there are no more pulse ends but there are more pulse starts
            next_event=next_pulse_end   # Next event must be a pulse start
        else:                           # If there are both more pulse starts and ends
            next_event = min([next_pulse_start, next_pulse_end])    # Next event is the one that comes sooner
        
        if next_event != -1:
            if t + h.dt > next_event: # If we would step over the next event
                h.dt = round(next_event - t, 9) # Set dt so that we will hit the event exactly
        from netpyne import sim
        if t + h.dt > sim.cfg.duration: # If we would step over the sim duration
            h.dt = round(sim.cfg.duration - t, 9) # Set dt so that we will end the simulation

    return wav, time, NetPyNE_interval_func


def check_nonspecific_parameters(
        simulation_duration_ms,
        efield_amplitude_V_per_m,
        stim_start_ms,
        stim_end_ms,
        pulse_width_ms,
        default_dt,
        pulse_dt,
        num_time_steps_in_pulse,
    ):
    simplefilter('default', UserWarning)
    # Check that the parameters which are not specific to stimulation type are valid
    if simulation_duration_ms <= 0:
        raise ValueError(f"simulation_duration_ms [{simulation_duration_ms}] must be > 0")
    if efield_amplitude_V_per_m == None:
        raise ValueError(f"efield_amplitude_V_per_m must be defined")
    if stim_start_ms < 0:
        raise ValueError(f"stim_start_ms [{stim_start_ms}] must be >= 0")
    if simulation_duration_ms <= stim_start_ms:
        warn(f"simulation_duration_ms [{simulation_duration_ms}] should be > stim_start_ms [{stim_start_ms}]")
    # Also set stim_end_ms if it is not set properly
    if stim_end_ms == None or stim_end_ms > simulation_duration_ms:
        stim_end_ms = simulation_duration_ms
    if stim_end_ms <= stim_start_ms:
        warn(f"stim_end_ms [{stim_end_ms}] should be > stim_start_ms [{stim_start_ms}]")
    if pulse_width_ms == None: # tDCS case
        pulse_width_ms = stim_start_ms-stim_end_ms
    if pulse_width_ms <= 0:
        raise ValueError(f"pulse_width_ms [{pulse_width_ms}] must be > 0")
    if default_dt <= 0:
        raise ValueError(f"default_dt [{default_dt}] must be > 0")
    if pulse_dt == None:
        pulse_dt = default_dt
    
    # Check and implement num_time_steps_in_pulse
    if num_time_steps_in_pulse != None:
        if num_time_steps_in_pulse <= 0:
            raise ValueError(f"num_time_steps_in_pulse [{num_time_steps_in_pulse}] must be > 0 or None")
        else:
            pulse_dt = min(pulse_dt, pulse_width_ms/num_time_steps_in_pulse)
    pulse_dt = min(pulse_dt, default_dt) # pulse_dt takes the smallest dt value between those defined by pulse_dt, default_dt, and num_time_steps_in_pulse
    if pulse_dt <= 0:
        raise ValueError(f"pulse_dt [{pulse_dt}] must be > 0 or None")
    return stim_end_ms, pulse_dt, pulse_width_ms
    

def plot_efield(wav, time):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("E-field Plots")
    ax[0].step(time, wav, where="post")
    ax[0].set_title("Raw Step Function")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("E-field (mV/um)")

    ax[1].plot(time, wav)
    ax[1].set_title("Interpolated Function")
    ax[1].set_xlabel("Time (ms)")

    # cur_t = 0
    # for start in pulse_start_times_ms:
        

    # num_sample_steps = ceil(time[-1] / dt)
    # effective_duration_ms = num_sample_steps * dt
    # num_sample_points = num_sample_steps + 1  # Number of time points within the duration
    # sampled_time = np.linspace(0, effective_duration_ms, num_sample_points, endpoint=False)
    # sampled_time = np.append(sampled_time, time[-1])

    # sampled_wav = np.interp(sampled_time, time, wav)

    # ax[2].step(sampled_time, sampled_wav, where="post")
    # ax[2].set_title(f"Sampled Step Function with dt={dt} ms")
    # ax[2].set_xlabel("Time (ms)")


def get_efield_sTMS(
        simulation_duration_ms: float,
        efield_amplitude_V_per_m: float,
        num_pulses_per_burst: int,
        stim_start_ms: float = 0.,
        default_dt: float = 25e-3,
        tms_pulse_shape: str = "Ideal_Sine",
        tms_pulse_width_ms: float = 100e-3,
        pulse_interval_within_burst_ms: float | None = None,
        pulse_onset_interval_within_burst_ms: float | None = None,
        pulse_freq_within_burst_Hz: float | None = None,
        pulse_dt: float | None = None,
        num_time_steps_in_pulse: int | None = None,
        plot: bool = False,
        **kwargs,
    ):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    default_dt: Temporal resolution of pulses in ms (should be <= within-pulse simulation dt)
    tms_pulse_shape: Qualitative description of TMS waveform (see Shape class)
    tms_pulse_width_ms: Period of TMS pulse in ms
    tms_pulse_burst_pattern: Qualitative description of stimulation pattern (see Pattern class)
    num_pulses_per_burst: Number of pulses in one burst of a pattern
    pulse_interval_within_burst_ms: Duration of interval between pulses in a burst in ms
    pulse_onset_interval_within_burst_ms: Duration of interval between onset of pulses in a burst in ms
    pulse_freq_within_burst_Hz: Frequency of pulse onsets in a burst in Hz
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms, pulse_dt, pulse_width_ms = check_nonspecific_parameters(
            simulation_duration_ms=simulation_duration_ms,
            efield_amplitude_V_per_m=efield_amplitude_V_per_m,
            stim_start_ms=stim_start_ms,
            stim_end_ms=None,
            pulse_width_ms=tms_pulse_width_ms,
            default_dt=default_dt,
            pulse_dt=pulse_dt,
            num_time_steps_in_pulse=num_time_steps_in_pulse,
        )

    wav, time, NetPyNE_interval_func = generate_efield(
        burst_freq_Hz=None,
        simulation_duration_ms=simulation_duration_ms,  
        default_dt=default_dt, 
        stim_start_ms=stim_start_ms,  
        stim_end_ms=stim_end_ms,
        total_num_tms_pulse_bursts=1,
        efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
        pat=Pattern(
            pulse_shape=Shape(shape=tms_pulse_shape, pulse_width_ms=pulse_width_ms),
            num_pulses_per_burst=num_pulses_per_burst,
            pulse_interval_within_burst_ms=pulse_interval_within_burst_ms,
            pulse_onset_interval_within_burst_ms=pulse_onset_interval_within_burst_ms,
            pulse_freq_within_burst_Hz=pulse_freq_within_burst_Hz,
            ),
        pulse_dt=pulse_dt,
        )

    if plot:
        plot_efield(wav, time)
    
    return wav, time, NetPyNE_interval_func
    

def get_efield_rTMS(
        simulation_duration_ms: float,
        efield_amplitude_V_per_m: float,
        burst_freq_Hz: float,
        stim_start_ms: float = 0.,
        stim_end_ms: float | None = None,
        total_num_tms_pulse_bursts: int | None = None,
        default_dt: float = 25e-3,
        tms_pulse_shape: str = "Ideal_Sine",
        tms_pulse_width_ms: float = 100e-3,
        num_pulses_per_burst: int | None = None,
        pulse_interval_within_burst_ms: float | None = None,
        pulse_onset_interval_within_burst_ms: float | None = None,
        pulse_freq_within_burst_Hz: float | None = None,
        pulse_dt: float | None = None,
        num_time_steps_in_pulse: int | None = None,
        plot: bool = False,
        **kwargs,
    ):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    total_num_tms_pulse_bursts: Total number of pulse bursts to include in time course
        Either stim_end_ms or total_num_tms_pulse_bursts will determine the number of pulse bursts based on which is more restrictive
    rtms_pulse_burst_freq_Hz: Frequency of rTMS pulse bursts
    default_dt: Temporal resolution of pulses in ms (should be <= within-pulse simulation dt)
    tms_pulse_shape: Qualitative description of TMS waveform (see Shape class)
    tms_pulse_width_ms: Period of TMS pulse in ms
    tms_pulse_burst_pattern: Qualitative description of stimulation pattern (see Pattern class)
    num_pulses_per_burst: Number of pulses in one burst of a pattern
    pulse_interval_within_burst_ms: Duration of interval between pulses in a burst in ms
    pulse_onset_interval_within_burst_ms: Duration of interval between onset of pulses in a burst in ms
    pulse_freq_within_burst_Hz: Frequency of pulse onsets in a burst in Hz
        Only one of pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, or pulse_freq_within_burst_Hz must be defined
        Highest priority when defined | pulse_interval_within_burst_ms > pulse_onset_interval_within_burst_ms > pulse_freq_within_burst_Hz | lowest priority
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms, pulse_dt, pulse_width_ms = check_nonspecific_parameters(
            simulation_duration_ms=simulation_duration_ms,
            efield_amplitude_V_per_m=efield_amplitude_V_per_m,
            stim_start_ms=stim_start_ms,
            stim_end_ms=stim_end_ms,
            pulse_width_ms=tms_pulse_width_ms,
            default_dt=default_dt,
            pulse_dt=pulse_dt,
            num_time_steps_in_pulse=num_time_steps_in_pulse,
        )

    # Check that the rTMS-specific parameters are valid
    if total_num_tms_pulse_bursts != None:
        if total_num_tms_pulse_bursts < 0:
            raise ValueError(f"total_num_tms_pulse_bursts [{total_num_tms_pulse_bursts}] must be >= 0")
    if burst_freq_Hz == None:
        raise ValueError(f"burst_freq_Hz must be defined")
    if burst_freq_Hz <= 0:
        raise ValueError(f"burst_freq_Hz [{burst_freq_Hz}] must be > 0")
    
    wav, time, NetPyNE_interval_func = generate_efield(
            burst_freq_Hz=burst_freq_Hz,
            simulation_duration_ms=simulation_duration_ms,  
            default_dt=default_dt, 
            stim_start_ms=stim_start_ms,  
            stim_end_ms=stim_end_ms,
            total_num_tms_pulse_bursts=total_num_tms_pulse_bursts,
            efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
            pat=Pattern(
                pulse_shape=Shape(shape=tms_pulse_shape, pulse_width_ms=tms_pulse_width_ms),
                num_pulses_per_burst=num_pulses_per_burst,
                pulse_interval_within_burst_ms=pulse_interval_within_burst_ms,
                pulse_onset_interval_within_burst_ms=pulse_onset_interval_within_burst_ms,
                pulse_freq_within_burst_Hz=pulse_freq_within_burst_Hz,
            ),
            pulse_dt=pulse_dt,
        )

    if plot:
        plot_efield(wav, time)
    
    return wav, time, NetPyNE_interval_func
    

def get_efield_tACS(
        simulation_duration_ms: float,
        efield_amplitude_V_per_m: float,
        tacs_freq_Hz: float,
        stim_start_ms: float = 0.,
        stim_end_ms: float | None = None,
        default_dt: float = 25e-3,
        pulse_dt: float | None = None,
        num_time_steps_in_pulse: int | None = None,
        plot: bool = False,
        **kwargs,
    ):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    default_dt: Temporal resolution of pulses in ms (should be <= simulation dt)
    tacs_freq_Hz: Frequency of tACS stimulation
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    # Check that the tACS-specific parameter is valid
    if tacs_freq_Hz <= 0:
        raise ValueError(f"tacs_freq_Hz [{tacs_freq_Hz}] must be > 0")
    
    pulse_width_ms = 1/tacs_freq_Hz * s
    
    stim_end_ms, pulse_dt, pulse_width_ms = check_nonspecific_parameters(
            simulation_duration_ms=simulation_duration_ms,
            efield_amplitude_V_per_m=efield_amplitude_V_per_m,
            stim_start_ms=stim_start_ms,
            stim_end_ms=stim_end_ms,
            pulse_width_ms=pulse_width_ms,
            default_dt=default_dt,
            pulse_dt=pulse_dt,
            num_time_steps_in_pulse=num_time_steps_in_pulse,
        )

    wav, time, NetPyNE_interval_func = generate_efield(
            burst_freq_Hz=tacs_freq_Hz,
            simulation_duration_ms=simulation_duration_ms,  
            default_dt=default_dt, 
            stim_start_ms=stim_start_ms,  
            stim_end_ms=stim_end_ms,
            total_num_tms_pulse_bursts=None,
            efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
            pat=Pattern(
                pulse_shape=Shape(shape="Ideal_Sine", pulse_width_ms=pulse_width_ms),
                num_pulses_per_burst=1,
            ),
            pulse_dt=pulse_dt,
        )

    if plot:
        plot_efield(wav, time)
    
    return wav, time, NetPyNE_interval_func
    

def get_efield_tDCS(
        simulation_duration_ms: float,
        efield_amplitude_V_per_m: float,
        stim_start_ms: float = 0.,
        stim_end_ms: float | None = None,
        default_dt: float = 25e-3,
        pulse_dt: float | None = None,
        num_time_steps_in_pulse: int | None = None,
        plot: bool = False,
        **kwargs,
    ):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    default_dt: Temporal resolution of pulses in ms (should be <= simulation dt) TODO: clarify purpose for tDCS
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """

    stim_end_ms, pulse_dt, pulse_width_ms = check_nonspecific_parameters(
            simulation_duration_ms=simulation_duration_ms,
            efield_amplitude_V_per_m=efield_amplitude_V_per_m,
            stim_start_ms=stim_start_ms,
            stim_end_ms=stim_end_ms,
            pulse_width_ms=None,
            default_dt=default_dt,
            pulse_dt=pulse_dt,
            num_time_steps_in_pulse=num_time_steps_in_pulse,
        )

    wav, time, NetPyNE_interval_func = generate_efield(
            burst_freq_Hz=None,
            simulation_duration_ms=simulation_duration_ms,  
            default_dt=default_dt, 
            stim_start_ms=stim_start_ms,  
            stim_end_ms=stim_end_ms,
            total_num_tms_pulse_bursts=1,
            efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
            pat=Pattern(
                pulse_shape=Shape(shape="Ideal_Square", pulse_width_ms=pulse_width_ms),
                num_pulses_per_burst=1,
            ),
            pulse_dt=pulse_dt,
        )

    if plot:
        plot_efield(wav, time)
    
    return wav, time, NetPyNE_interval_func


def get_efield(stim_type: str, **kwargs):
    if stim_type == 'sTMS':
        return get_efield_sTMS(**kwargs)
    elif stim_type == 'rTMS':
        return get_efield_rTMS(**kwargs)
    elif stim_type == 'tACS':
        return get_efield_tACS(**kwargs)
    elif stim_type == 'tDCS':
        return get_efield_tDCS(**kwargs)