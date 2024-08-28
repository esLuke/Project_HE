import os
import time
import inspect, copy
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from astropy import units as u
from astropy.constants import G, c, M_sun, R_sun, L_sun, au, pc

import lal, lalsimulation
import pycwb
import pycbc
import pycbc.scheme as _scheme
from pycbc import pnutils
from pycbc.types import TimeSeries, FrequencySeries, zeros, Array
from pycbc.types import real_same_precision_as, complex_same_precision_as
from pycbc.fft import fft
from pycbc.filter import matched_filter
from pycbc.filter import highpass
from pycbc.filter import interpolate_complex_frequency, resample_to_delta_t
from pycbc.waveform import parameters
from pycbc.waveform import get_td_waveform
from pycbc.waveform import td_approximants, fd_approximants
from pycbc.waveform import utils as wfutils
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.psd.analytical import EinsteinTelescopeP1600143
from pycbc.noise import noise_from_psd
from pycbc.filter import matched_filter
from pycbc.filter import highpass


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

@dataclass
class HyperbolicGW:
    m1: float
    m2: float
    hyp_ecc: float
    b: float
    distance: float
    inclination: float
    deltaT: float = 1.0 / 4096
    f_min: float = 20
    f_ref: float = 20
    _args: dict = field(init=False, repr=False)
    
    def __post_init__(self):
        self._args = {
            "m1": self.m1 * lal.MSUN_SI,
            "m2": self.m2 * lal.MSUN_SI,
            "s1x": 0, "s1y": 0, "s1z": 0, 
            "s2x": 0, "s2y": 0, "s2z": 0, 
            "distance": self.distance * lal.PC_SI,
            "inclination": self.inclination,
            "phiRef": 0,
            "longAscNodes": 0,
            "eccentricity": 0,
            "meanPerAno": 0,
            "deltaT": self.deltaT,
            "f_min": self.f_min,
            "f_ref": self.f_ref,
            "params": lal.CreateDict(),
            "approximant": None
       }
        self.M = self._args["m1"] + self._args["m2"]
        self.b_SI = self.M * lal.G_SI / lal.C_SI ** 2
        self.b = self.b * self.b_SI
        self.e = self.hyp_ecc
        
        self.psd_type = "aligo" 
        self.seed = 333
        

    @property
    def args(self):
        return self._args

    def _hyp_params(self):
        self._args["approximant"] = lalsimulation.GetApproximantFromString("HyperbolicTD")
        params = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertHyperbolicEccentricity(params, self.e)
        lalsimulation.SimInspiralWaveformParamsInsertImpactParameter(params, self.b)
        self._args["params"] = params

    def _hyp_generator(self, domain):
        self._hyp_params() 
        if domain == 'td':
            generator = lalsimulation.SimInspiralChooseTDWaveform
        elif domain == 'fd':
            generator = lalsimulation.SimInspiralChooseFDWaveform
        else:
            raise ValueError("Invalid domain: choose 'td' for time-domain or 'fd' for frequency-domain")
           
        hp, hc = generator(*list(self._args.values()))
        
        return hp, hc
        
    def td_waveform(self):
        hp, hc = self._hyp_generator(domain='td')
         # time series
        hp_ts = TimeSeries(hp.data.data[:], delta_t=hp.deltaT, epoch=hp.epoch)
        hc_ts = TimeSeries(hc.data.data[:], delta_t=hc.deltaT, epoch=hc.epoch)
        return hp_ts, hc_ts

    def _convert_to_fd(self, hpt, hct):
        hpf = FrequencySeries(np.zeros(len(hpt)//2 + 1, dtype=np.complex128), delta_f=1.0/hpt.duration)
        hcf = FrequencySeries(np.zeros(len(hct)//2 + 1, dtype=np.complex128), delta_f=1.0/hct.duration)
        fft(hpt, hpf)
        fft(hct, hcf)
        return hpf, hcf
        
    def fd_waveform(self):
        hp_fs, hc_fs = self._convert_to_fd(*(self.td_waveform()))
        return hp_fs, hc_fs

    def _hyp_filter(self, domain):
        hp_ts_fir, hc_ts_fir = highpass(self.td_waveform()[0], 10.0), highpass(self.td_waveform()[1], 10.0)
        if domain == 'td':
            return hp_ts_fir, hc_ts_fir
        elif domain == 'fd':
            hp_fs_fir, hc_fs_fir = self._convert_to_fd(hp_ts_fir, hc_ts_fir)
            return hp_fs_fir, hc_fs_fir
        else:
            raise ValueError("Invalid domain: choose 'td' for time-domain or 'fd' for frequency-domain")
        

    def hyp_psd(self):
        hp_fs_fir, hc_fs_fir = self._hyp_filter('fd')
        if self.psd_type == "aligo":
            psd = aLIGOZeroDetHighPower(len(hp_fs_fir), delta_f=hp_fs_fir.delta_f, low_freq_cutoff=10.0)
        elif self.psd_type == "et":
            psd_path = os.path.join(project_root, 'src', 'ET_D_sum_psd.txt')
            psd = pycbc.psd.from_txt(psd_path, len(hp_fs_fir), delta_f=hp_fs_fir.delta_f, 
                             low_freq_cutoff=10.0, is_asd_file=True)
        elif self.psd_type == "ce":
            psd_path = os.path.join(project_root, 'src', 'CE_2_psd.txt')
            psd = pycbc.psd.from_txt(psd_path, len(hp_fs_fir), delta_f=hp_fs_fir.delta_f, 
                            low_freq_cutoff=10.0, is_asd_file=True)
        else:
            raise ValueError("Invalid PSD type specified.")
        return psd
        
    def hyp_noise(self):
        hp_ts_fir, hc_ts_fir = self._hyp_filter('td')
        psd = self.hyp_psd()
        noise = noise_from_psd(len(hp_ts_fir), hp_ts_fir.delta_t, psd, seed=self.seed)
        noise_ts = TimeSeries(noise.data, delta_t=noise.delta_t, epoch=hp_ts_fir.start_time)
        noise_fs = noise_ts.to_frequencyseries()
        return noise_ts, noise_fs

    def hyp_snr(self):
        try:
            hp_fs_fir, hc_fs_fir = self._hyp_filter('fd')
            noise_ts, noise_fs = self.hyp_noise()
            signal_fs = hp_fs_fir + noise_fs
            psd = self.hyp_psd()
            snr = matched_filter(hp_fs_fir, signal_fs, psd=psd, low_frequency_cutoff=10.0)
            max_snr = abs(snr).max()
            return max_snr
        except RuntimeError as err:
            print(f"Error occurred for b={self.b/self.b_SI}, e={self.e}: {err}")
            return None


class detector_ALIGO(HyperbolicGW):
    def __init__(self, m1, m2, hyp_ecc, b, distance, inclination):
        super().__init__(m1, m2, hyp_ecc, b, distance, inclination)
        self.psd_type = "aligo"  # default

class detector_ETD(HyperbolicGW):
    def __init__(self, m1, m2, hyp_ecc, b, distance, inclination):
        super().__init__(m1, m2, hyp_ecc, b, distance, inclination)
        self.psd_type = "et" 

class detector_CE2(HyperbolicGW):
    def __init__(self, m1, m2, hyp_ecc, b, distance, inclination):
        super().__init__(m1, m2, hyp_ecc, b, distance, inclination)
        self.psd_type = "ce" 

class HyperAnalysis(HyperbolicGW):
    def __init__(self, m1, m2, hyp_ecc, b, distance, inclination, 
                 detector, seed):
        super().__init__(m1, m2, hyp_ecc, b, distance, inclination)
        self.b_min = 30
        self.b_max = 70
        self.e_min = 1.05
        self.e_max = 1.5
        self.psd_type = detector
        self.seed = seed


def snr_b(b_values, detector, ecc=1.1):
    snr_b = []
    valid_b = []
    error_b = []
    if detector == "aligo":
        detector_class = detector_ALIGO
    elif detector == "et":
        detector_class = detector_ETD
    elif detector == "ce":
        detector_class = detector_CE2
    else:
        raise ValueError("Invalid detector specified. Use 'aligo', 'et' or 'ce'.")
        
    start_time = time.time()
    for i, b in enumerate(b_values):
        test = detector_class(10, 10, ecc, b, 1.6e6, np.pi/3)
        snr = test.hyp_snr()
        if snr is not None:
            snr_b.append(snr)
            valid_b.append(b)
        else:
            error_b.append(b)
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            estimated_time_remaining = (len(b_values) - (i + 1)) * (elapsed_time / (i + 1))
            print(f"Progress: {i + 1} out of {len(b_values)} b values calculated.")
            print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
    return snr_b, valid_b, error_b


def snr_e(e_values, detector, b=100):
    snr_e = []
    valid_e = []
    error_e = []
    if detector == "aligo":
        detector_class = detector_ALIGO
    elif detector == "et":
        detector_class = detector_ETD
    elif detector == "ce":
        detector_class = detector_CE2
    else:
        raise ValueError("Invalid detector specified. Use 'aligo', 'et' or 'ce'.")

    start_time = time.time()
    for i, e in enumerate(e_values):
        test = detector_class(10, 10, e, 100, 1.6e6, np.pi/3)
        snr = test.hyp_snr()

        if snr is not None:
            snr_e.append(snr)
            valid_e.append(e)
        else:
            error_e.append(e)
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            estimated_time_remaining = (len(e_values) - (i + 1)) * (elapsed_time / (i + 1))
            print(f"Progress: {i + 1} out of {len(e_values)} ecc values calculated.")
            print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
    return snr_e, valid_e, error_e


def calculate_snr_b_sd(detector, b, seed_list, e=1.1):
    snr_values = []
    for seed in seed_list:
        test = HyperAnalysis(10, 10, 1.1, b, 1.6e6, np.pi/3, detector, seed)
        snr = test.hyp_snr() 
        if snr is not None:
            snr_values.append(snr)
    
    if len(snr_values) == 0:
        print(f"No valid SNR values for b = {b}. Skipping this b value.")
        return None, None
    
    snr_mean = np.mean(snr_values)
    snr_sd = np.std(snr_values)
    
    return snr_mean, snr_sd

def wrapper_snr_b_sd(detector, b_list, seed_list, e=1.1):
    snr_b_mean = []
    snr_b_sd = []
    valid_b = [] 

    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(calculate_snr_b_sd, detector, b, seed_list, e): b for b in b_list}
        
        for i, future in enumerate(as_completed(futures)):
            b = futures[future]
            try:
                snr_mean, snr_sd = future.result()
                if snr_mean is not None:
                    snr_b_mean.append(snr_mean)
                    snr_b_sd.append(snr_sd)
                    valid_b.append(b)
            except Exception as exc:
                print(f"b = {b} generated an exception: {exc}")

            print(f"Progress: {i + 1} out of {len(b_list)} b values processed.")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    return valid_b, snr_b_mean, snr_b_sd

def snr_e_sd(detector, e_list, seed_list, b=100):
    snr_e_mean = []
    snr_e_sd = []
    for e in e_list:
        snr_values = []
        for seed in seed_list:
            test = HyperAnalysis(10, 10, e, 100, 1.6e6, np.pi/3, detector, seed)
            snr = test.hyp_snr()         
            snr_values.append(snr)
            
        snr_mean = np.mean(snr_values)
        snr_e_mean.append(snr_mean)
        snr_sd = np.std(snr_values)
        snr_e_sd.append(snr_sd)

    return snr_e_mean, snr_e_sd


def snr_be(b, e):
    test = HyperbolicGW(10, 10, e, b, 1.6e6, np.pi/3)
    snr = test.hyp_snr()
    return snr, b, e

def wrapper_snr_be(params):
    return snr_be(*params)