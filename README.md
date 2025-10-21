# Circulation-Benchmarking-AI-Emulators
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 

Code and link to data to reproduce diagnostics and figures for submitted manuscript, "Benchmarking atmospheric circulation variability in an AI emulator, ACE2, and a hybrid model, NeuralGCM"

Python3.12 and Bash used to run code:
<p align="left">
  <a href="https://www.python.org/">
    <img src="https://skillicons.dev/icons?i=python,bash" />
  </a>
</p>

Preprint Available on ArXiv:
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://www.arxiv.org/abs/2510.04466)

Link to Zenodo repository with data: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17351379.svg)](https://doi.org/10.5281/zenodo.17351379)


We present 4 benchmarking metrics evaluating the capabilities of an AI Emulator (ACE2-ERA5) and a Hybrid AI-Atmospheric Model (NeuralGCM) to capture atmopsheric circulation variability.

# 1. Quasi-Biennial Oscillation (QBO)


# 2. Convectively coupled equatorial waves (CCWs)
In the tropics, atmospheric circulation and its associated impacts are generally determined by atmospheric waves coupled to deep convection ([Kiladis et al., 2009](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008RG000266)); however, this convection that is normally parameterized in physics-based models (often poorly) is implicitly learned by the neural networks in the AI emulator and hybrid model. Specific regions of the wavenumber-frequency spectra have been associated with leading modes of tropical variability, such as the MJO and equatorial Rossby, Kelvin, Mixed Rossby gravity (MRG), and inertio-gravity (IG) waves. To test the representation of convectively coupled waves, we compute frequency-wavenumber spectra following [Wheeler and Kiladis (1999)](https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml), using daily precipitation from ERA5, AMIP, and ACE2-ERA5. This ver sion of NeuralGCM does not prognostically forecast precipitation directly, but it is in ferred from total column moisture convergence (i.e., precipitation minus evaporation).

To evaluate convectively coupled waves in the tropics, we follow the methodology from [Wheeler and Kiladis (1999)](https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml). The code uses functions from Brian Medeiros' Github to compute wavenumber-frequency spectra [https://github.com/brianpm/wavenumber_frequency](https://github.com/brianpm/wavenumber_frequency).



# 3. Extratropical eddy-mean flow interactions


# 4. Poleward propagation of the Southern Annular Mode (SAM)
