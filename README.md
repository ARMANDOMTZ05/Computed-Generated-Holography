# Computed Generated Holography
## Description
Code for generation of binary amplitude hologram based on the K. Mithcell, et al. (2016) using the Gerchberg–Saxton algorithm for phase retrieval

### Gerchberg–Saxton pseudocode algorithm


```
A = ifft(Target)
for x in Iterations
    B = Amplitude(Source) x exp(j x Phase(A))
    C = fft(B)
    D = Amplitude(Target) x exp(j x Phase(C))
    A = ifft(D)

Retrieved_Phase = Phase(A)
```


### Binarization

$T(x, y) = \frac{1}{2} + \frac{1}{2} \text{sgn} \left[\cos(p(x, y)) + \cos(q(x, y))\right]$


## Dependencies
For the creation of the Hologram it is used [PyTorch](https://pytorch.org/get-started/locally/) supported by a GPU to reduce the processing time.

For Volumetric Additive Manufacturing (VAM) the [VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox/tree/v0.1.5b) library by Joseph Toombs is used for generating the projections.

## Disclaimer
The code for the VAM projections was generated based on the VAMToolbox with the modification of the position of the sinogram. Instead of centering the projection it displaces it to a corner. This reduces the resolution four times

## References
[1] R. W. Gerchberg and W. O. Saxton, “A practical algorithm for the determination of the phase from image and diﬀraction plane pictures”, Optik 35, 237 (1972).

[2] K. Mitchell, S. Turtaev, M. Padgett, T. Cizmár, and D. Phillips, “High-speed spatial control of the intensity, phase and polarisation of vector beams using a digital micro-mirror device”, Opt. Express 24, 29269-29282 (2016).

[3] Forbes A. 2014, Laser Beam Propagation: Generation and Propagation of Customized Light (London: Taylor and Francis).