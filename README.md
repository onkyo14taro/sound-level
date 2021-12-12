# Sound level module

This module provides functions to calculate equivalent continuous sound level (L_eq) and percentile sound pressure level (L_x). The level calculation uses digital filters of A, B, C, and Z-weighting [1, 2]. They are designed by bilinear transform with prewarping [3].


## Usage

```python
from soundlevel import level_metrics, equivalent_level, percentile_level
import numpy as np

fs = 48000  # Sampling frequency [Hz]
wave = np.random.randn(100000)  # An audio wave

# You can choose between A, B, C, and Z-weighting.
# Calculate equivalent continuous sound level (L_eq).
L_Beq = equivalent_level(wave, fs, weighting='B')
# Calculate percentile sound pressure level (L_x).
L_Cx = percentile_level(wave, fs, [90, 50, 10], weighting='C')  # L_90, L_50, L_10
# Short-hand function to calculate L_eq, L_5, L_50, L_95.
L_Aeq, L_A5, L_A50, L_A95 = level_metrics(wave, fs, weighting='A')
```


## Notes

Digital filters can’t completely reproduce the ideal frequency responce of analogue filter.
In particular, a high sampling frequency is required to accurately approximate the frequency response in the high frequency range.
To satisfy standards (ANSI [1] and IEC[2]), **the sampling frequency is resampled to 96 kHz during processing by default**.
The sampling frequency during processing can be changed, but if the standard is not satisfied, an exception or warning will be raised.

```python
level_metrics(
    wave,
    fs,
    # Sampling frequency during processing
    fs_weighting=96_000.0,
    # Standard referenced (ANSI S.1-43 Type 0 is the strictest)
    standard='ANSI0',
    # If `True`, raise an error when `fs_weighting` does not satisfy `standard`.
    # If `False`, raise a warning.
    must_satisfy_standard=True,
)
```


## References

[1] American National Standards Institute, “ANSI S1.43: Specifications for Integrating-averaging Sound Level Meters,” Acoustical Society of America, 1997.  
[2] International Electrotechnical Commission, “Electroacoustics: Sound Level Meters. Specifications. Part 1,” International Electrotechnical Commission, 2013.  
[3] A. N. Rimell, N. J. Mansfield, and G. S. Paddan, “Design of digital filters for frequency weightings (A and C) required for risk assessments of workers exposed to noise,” Ind. Health, pp. 3–2013, 2014.