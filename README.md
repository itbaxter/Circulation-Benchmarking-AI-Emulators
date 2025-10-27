# Circulation-Benchmarking-AI-Emulators
Preprint Available on ArXiv:
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://www.arxiv.org/abs/2510.04466)

Link to Zenodo repository with data: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17351379.svg)](https://doi.org/10.5281/zenodo.17351379)

Code and link to data to reproduce diagnostics and figures for submitted manuscript, "Benchmarking atmospheric circulation variability in an AI emulator, ACE2, and a hybrid model, NeuralGCM"

Python3 and Bash used to run code:
<p align="left">
  <a href="https://www.python.org/">
    <img src="https://skillicons.dev/icons?i=python,bash" />
  </a>
</p>

## Quick run Guide
To recreate the figures first download and unzip the input data files from the Zenodo archive.

Then run bash script: "./scripts/run_scripts.sh <path/ to/ zenodo/ directory>". 

This will put all figures (in png files) in the plots directory.

## Metrics
We present 4 benchmarking metrics evaluating the capabilities of an AI Emulator (ACE2-ERA5) and a Hybrid AI-Atmospheric Model (NeuralGCM) to capture atmopsheric circulation variability.

## 1. [Quasi-Biennial Oscillation (QBO)](scripts/QBO)

![alt text][QBO]

The QBO, is characterized by the downward propagation of successive westerly and easterly winds with an average period of ∼ 28 months ([Baldwin et al., 2001](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/1999rg000073); [Anstey et al., 2022](https://www.nature.com/articles/s43017-022-00323-7)). In this study, the QBO index is defined as the monthly and latitude-weighted (10◦S to 10◦N ) mean zonal winds at 50 hPa. QBO amplitude is computed following [Richter et al. 2020](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD032362).

## 2. [Convectively coupled equatorial waves (CCWs)](scripts/WK99)
In the tropics, atmospheric circulation and its associated impacts are generally determined by atmospheric waves coupled to deep convection ([Kiladis et al., 2009](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008RG000266)); however, this convection that is normally parameterized in physics-based models (often poorly) is implicitly learned by the neural networks in the AI emulator and hybrid model. Specific regions of the wavenumber-frequency spectra have been associated with leading modes of tropical variability, such as the MJO and equatorial Rossby, Kelvin, Mixed Rossby gravity (MRG), and inertio-gravity (IG) waves. To test the representation of convectively coupled waves, we compute frequency-wavenumber spectra following [Wheeler and Kiladis (1999)](https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml), using daily precipitation from ERA5, AMIP, and ACE2-ERA5. This version of NeuralGCM does not prognostically forecast precipitation directly, but it is in ferred from total column moisture convergence (i.e., precipitation minus evaporation).

To evaluate convectively coupled waves in the tropics, we follow the methodology from [Wheeler and Kiladis (1999)](https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml). The code uses functions from Brian Medeiros' Github to compute wavenumber-frequency spectra [https://github.com/brianpm/wavenumber_frequency](https://github.com/brianpm/wavenumber_frequency).

## 3. [Extratropical eddy-mean flow interactions](scripts/RH91)
Extratropical atmospheric dynamics is dominated by the interactions of eddies (deviations from the zonal mean) and the jet stream (mean flow). This interaction can be quantified by eddy flux co-spectra, which are here computed following previous work (Hayashi, 1971; Randel & Held, 1991; Chen & Held, 2007; Lutsko et al., 2017). The code here builds on Matlab code from [Gang Chen](https://www.gchenpu.com/files/co-spectra/) and [Nick Lutsko](https://nicklutsko.github.io/code/). 

## 4. [Poleward propagation of the Southern Annular Mode (SAM)](scripts/LH23)
The large-scale circulation in the Southern Hemisphere is dominated by the Southern Annular Mode (SAM), which has a 150-day periodicity ([Lubis & Hassanzadeh, 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022AV000833)). The timescale of the annular mode is determined by wave-mean flow interactions in the extratropics. Eddy feedbacks and their timescales has been used to evaluate the variability of physics-based models ([Gerber et al., 2008](agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008GL035712)). Here, we follow the approach in [Lubis & Hassanzadeh (2023)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022AV000833) to compute the propagation of SAM, using daily mean horizontal wind data at 500 hPa from ERA5, AMIP, ACE2-ERA5, and NeuralGCM from January 1981 to December 2014. NCL scripts are used to compute EOF following code from [Sandro Lubis](https://zenodo.org/records/7916770).



[QBO]: https://github.com/itbaxter/Circulation-Benchmarking-AI-Emulators/plots/qbo_time_series-2member.png "QBO Time Series"
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
