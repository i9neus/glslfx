bool IteratePolynomialRegression(inout vec4 rgba, ivec2 xy)
{        
    float gaussGain = pow(kGaussianDecay, float(gBandIdx));
    
    int frame0 = (iFrame % 2), frame1 = (iFrame + 1) % 2;
    if(gFrameIdx == frame1) 
    { 
        rgba = texelFetch(iChannel1, xy, 0);
        return true; 
    }    
    
    int x = gProbeIdx % kProbeCols;
    int y = gProbeIdx / kProbeCols;  
    
    vec4 thisTexel = texelFetch(iChannel0, getFourierCoord(x, y), 0);
    if(thisTexel.w == -1.0) { rgba = vec4(0.0, 0.0, 0.0, 0.0); return false; }      
     
    vec3 rgb = vec3(0.0);
    float sumW = 0.0;
    
    if(kNlm) { initPatch(x, y, 1); }
    
    for(int chnl = 0; chnl < 3; chnl++)
    {
        float maxP = -kFltMax, minP = kFltMax;
        float p[kKernelArea];
        float w[kKernelArea];
        
        for(int v = -kKernelRadius, idx = 0; v <= kKernelRadius; ++v)
        {
            for(int u = -kKernelRadius; u <= kKernelRadius; ++u, ++idx)
            {            
                p[idx] = w[idx] = 0.0;         
                
                //if(x+u < 0 || x+u >= kProbeCols || y+v < 0 || y+v >= kProbeRows) { continue; }
                
                vec4 texel = texelFetch(iChannel0, getFourierCoord(x+u, y+v), 0);
                texel.xyz /= max(1.0, texel.w - 1.0);
                
                float dist = length(vec2(u, v)) / (float(kKernelRadius) + 0.5);
                float weight = 1.0;
                if(u != 0 || v != 0)
                {
                    //weight = exp(-sqr(dist * 2.0 * gaussGain));
                    weight = max(0.0, 1.0 - sqr(dist) * gaussGain);
                    if(kNlm) 
                    { 
                        weight *= getPatchDistance(x+u, y+v, chnl, 1);  
                    }   
                }
                
                p[idx] = texel[chnl];
                w[idx] = weight;
                sumW += weight;
                maxP = max(maxP, p[idx]);
                minP = min(minP, p[idx]);
            }
        }         
        
        maxP = minP + max(1e-5, maxP - minP); 
       
        for(int idx = 0; idx < kKernelArea; idx++) { p[idx] = (p[idx] - minP) / (maxP - minP); } // Normalise inputs   
        
        float C[kNumBases]; 
        int blockIdx = kSectorBlockSize * gSectorIdx +       // Sector
                       kBandBlockSize +                      // Fourier coeffs
                       frame1 * kBandBlockSize * kNumBases + // Frame
                       gBandIdx * kNumBases;                 // Band

        for(int t = 0; t < kNumBases; t++)
        {
            vec4 texel = texelFetch(iChannel1, ivec2(blockIdx + t, gProbeIdx), 0);
            if(thisTexel.w == 0.0) { texel = rand(); }           
         
            C[t] = texel[chnl];
        }
        
        float L = 0.0; // Loss function
        float dLdC[kNumBases]; // Gradient
        for(int it = 0; it < kNumIterations; ++it)
        {
            L = 0.0; 
            for(int t = 0; t < kNumBases; t++) { dLdC[t] = 0.0; }  
            for(int v = -kKernelRadius, idx = 0; v <= kKernelRadius; ++v)
            {
                for(int u = -kKernelRadius; u <= kKernelRadius; ++u, ++idx)
                { 
                    if(w[idx] == 0.0) { continue; }                    
                    float x = float(u) / float(kKernelRadius), y = float(v) / float(kKernelRadius);
                    
                    // Monomial matrix
                    float D[kNumBases];                    
                    if(kConstantRegression)
                    {
                        D[0] = 1.0;
                        for(int t = 1; t < kNumBases; ++t) { D[t] = 0.0; }
                    }
                    else
                    { 
                        float xExp = 1.0;
                        for(int xt = 0, t = 0; xt < kPolyOrder; xt++)
                        {                        
                            float yExp = 1.0;
                            for(int yt = 0; yt < kPolyOrder; yt++, t++)
                            {                           
                                D[t] = xExp * yExp;
                                yExp *= y;
                            }
                            xExp *= x;
                        }                   
                    }
                     
                    float sigma = -p[idx];
                    for(int t = 0; t < kNumBases; t++) { sigma += C[t] * D[t]; }
                    
                    for(int t = 0; t < kNumBases; t++)
                    {
                        dLdC[t] += 2.0 * D[t] * sigma * w[idx];
                    }
                    L += sqr(sigma) * w[idx];
                }
            }
            L /= sumW;
            
            // Gradient descent
            for(int t = 0; t < kNumBases; t++) 
            { 
                dLdC[t] /= sumW;
                C[t] -= kLearningRate * dLdC[t] / max(L, 1e-2); 
                
                C[t] = clamp(C[t], 0.0, 1.0);
            }
        }
        
        rgba[chnl] = C[gBasisIdx];
    }    
    
    rgba.w += 1.0;
    
    return true;
}