![glslfx thumbnail gallery](https://github.com/i9neus/glslfx/blob/master/examples/images/thumbnails.jpg "glslfx thumbnail gallery")

# glslfx #

glslfx is a self-contained graphics library for creating dynamic visual effects in GLSL shaders. The project grew from my collection of generative art projects, many of which came to evolve from a common code library. This repository is an effort to de-duplicate, consolidate and clean up the code, and to make it available under a public license.

## Features ##

- 3D
  - Physically based rendering effects using wavefront Monte Carlo path tracing.
  - Ray/object intersectors including KIFS fractals, metaballs, SDFs, and implicit primitives.
  - Library of BRDF/BSDFs for surface and volume shading.
  - Light samplers for next event estimation.
  - Orthographic, thin-lens and fish-eye cameras.
  - Spectral sampling.
  - Delta tracking for volume rendering

- 2D
  - Camera effects including glare and diffraction bloom, depth-of-field blur, film grain and vignetting.
  - Procedural texture generators.
  - In-shader image decompression.
  - Colour grading.

- Math
  - 2D and 3D parametric curve evaluation including tabulated arc length approximation and samplers.
  - Utility functions for spherical harmonics, Fourier and wavelet basis functions.
  - Pseudo- and quasi-random number generation and hash functions.
  - Artificial neural network training and evaluation.
  - Function fitting using gradient descent.
  - Eigendecomposion and principle components analysis.
  - Entropy coding/decoding.
  - Colour space transformations.

- Utilities
  - svg2glsl: Allows spline curves to be embedded in the shader.
  - png2glsl: Image compressor and converter.

## Usage and Documentation ##
 
glslfx is still very early on in development. For now, the code is supplied "as is"; documentation, tutorials and guidance will be added as I clean everything up. For examples of the code running in-place, see my Shadertoy page at https://www.shadertoy.com/user/igneus
