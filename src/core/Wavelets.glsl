/*//////////////////////////////////////////////////////////////////////////////////////////////////

   Copyright (c) 2021 Ben Spencer   
   Released under the MIT Licence

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
    
   Description:
   
   This file is responsible for the texture decompression and internal state. The blocks of data
   in the buffers below contain three representations of the input image at different resolutions.
   Which resolution the decoder uses can be selected by modifying the kTextureResolution macro in
   the Common source file. Larger images mean a slower compilation time than smaller ones. Older systems
   are likely to struggle with the highest resolutions, so choose level 0 if compilation takes a long time

//////////////////////////////////////////////////////////////////////////////////////////////////*/

// Bi-orthogonal Cohen–Daubechies–Feauveau wavelet coefficients
// https://en.wikipedia.org/wiki/Cohen-Daubechies-Feauveau_wavelet
// NOTE: The coefficients in the Wikipedia page have the wrong offsets
const float[] cdf97InverseFather = float[](0.0, 
                                     -0.091271763114 * 0.5, 
                                     -0.057543526229 * 0.5, 
                                     0.591271763114 * 0.5,
                                     1.11508705 * 0.5,
                                     0.591271763114 * 0.5,
                                     -0.057543526229 * 0.5,
                                     -0.091271763114 * 0.5,
                                     0.0,
                                     0.0);
                                     
const float[] cdf97InverseMother = float[](0.0,
                                     0.026748757411 * 0.5,
                                     0.016864118443 * 0.5,
                                     -0.078223266529 * 0.5,
                                     -0.266864118443 * 0.5,
                                     0.602949018236 * 0.5,
                                     -0.266864118443 * 0.5,
                                     -0.078223266529 * 0.5,
                                     0.016864118443 * 0.5,
                                     0.026748757411 * 0.5);


// Orthogonal Haar wavelet coefficients
// https://en.wikipedia.org/wiki/Haar_wavelet
const float[] haarInverseFather = float[](0.5, 0.5);
const float[] haarInverseMother = float[](0.5, -0.5);

// Discrete inverse wavelet transform for the Haar basis
// https://en.wikipedia.org/wiki/Discrete_wavelet_transform
float InverseHaarTransform(ivec2 xy, ivec2 basis, ivec2 sourceOffset, int size, sampler2D sampler)
{
    int halfSize = size / 2;
    ivec2 xyLow = xy - basis * (xy + ivec2(1, 1)) / 2; 
    ivec2 xyHigh = xyLow + basis * halfSize;
    int pairIdx = sum((xyLow + xyLow * basis) - xy);
    
    float r;
    if(pairIdx == 0)
    {
        r = haarInverseFather[0] * texelFetch(sampler, xyLow + sourceOffset, 0).x +
            haarInverseMother[0] * texelFetch(sampler, xyHigh + sourceOffset, 0).x;
    }
    else
    {
        r = haarInverseFather[1] * texelFetch(sampler, xyLow + sourceOffset, 0).x +
            haarInverseMother[1] * texelFetch(sampler, xyHigh + sourceOffset, 0).x;
    }
    
    return r * 2.0;
}

// Discrete inverse wavelet transform using the Cohen–Daubechies–Feauveau basis
float InverseCDF97Transform(ivec2 xy, ivec2 basis, ivec2 sourceOffset, int size, sampler2D sampler)
{
    int halfSize = size / 2;
    ivec2 xyLow = xy - basis * (xy + ivec2(1, 1)) >> 1; 
    ivec2 xyHigh = xyLow + basis * halfSize;       
    ivec2 origin = (basis.x == 1) ? ivec2(0, xy.y) : ivec2(xy.x , 0); 
    int pairIdx = (basis.x == 1) ? (xy.x & 1) : (xy.y & 1);    
    int i = (basis.x == 1) ? (xy.x >> 1) : (xy.y >> 1);
    
    float r = 0.0;     
    for (int b = -2; b <= 2; b++)
    {       
        int bFather = i + b;
        int bMother = bFather;
        if (bFather < 0) 
        { 
            bFather = -bFather; 
            bMother = -bMother - 1;
        }
        else if (bFather > halfSize - 1) 
        { 
            bFather = 1 + 2 * (halfSize - 1) - bFather; 
            bMother = 2 * (halfSize - 1) - bMother;
        }
     
        int k0 = -(2 * b) + 4;
        int k1 = k0 + 1;
        if (pairIdx == 0 && k0 >= 0)
        {
            r += cdf97InverseFather[k0] * texelFetch(sampler, origin + basis * bFather + sourceOffset, 0).x + 
                 cdf97InverseMother[k0] * texelFetch(sampler, origin + basis * (halfSize + bMother) + sourceOffset, 0).x;
        }
        else if (pairIdx == 1 && k1 < 10)
        {
            r += cdf97InverseFather[k1] * texelFetch(sampler, origin + basis * bFather + sourceOffset, 0).x + 
                 cdf97InverseMother[k1] * texelFetch(sampler, origin + basis * (halfSize + bMother) + sourceOffset, 0).x;
        }
    }

    return r * 2.0;
}


// The lowest levels of coefficients are transformed with the Haar wavelet to avoid having to deal with 
// doubly reflected boundaries
float InverseWaveletTransform(ivec2 xy, ivec2 basis, ivec2 sourceOffset, int size, sampler2D sampler)
{
    return (size <= 4) ? inverseHaarTransform(xy, basis, sourceOffset, size, sampler) :
                         inverseCDF97Transform(xy, basis, sourceOffset, size, sampler);
}