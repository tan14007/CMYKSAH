# Structure-Aware Halftoning 
Implementation of Pang et al. (2008) in Python with CMYK color space. The implementation based on https://github.com/cache-tlb/halftone C++ implementation.

## Difference from the paper
* Use Ostromoukhov (2001) initialization instead of error diffusion stated in the paper
* Different weighting factors for tone and structure similarity

## WIP
* [ ] Refactoring the code
* [ ] Use other similarity measurement rather than SSIM since it defined on grayscale image

