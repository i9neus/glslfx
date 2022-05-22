vec3 RenderLoadingIcon(in vec2 xy)
{   
    #define kLoadingInnerRadius       0.05            // The inner radius of the loading icon
    #define kLoadingOuterRadius       0.07            // The outer radius of the loading icon
    #define kLoadingAASamples         3               // The number of anti-aliasing samples of the loading icon

    #define kLoadingMidpoint     (0.5 * (kLoadingOuterRadius + kLoadingInnerRadius))
    #define kLoadingThickness    (0.5 * (kLoadingOuterRadius - kLoadingInnerRadius))
    #define kLoadingThicknessSqr (kLoadingThickness*kLoadingThickness)
    #define kLoadingCycle        20
    
    // Exclude anything outside the outer radius
    if(length2(xyToUv(xy, iResolution).xy - vec2(0.5)) > sqr(kLoadingOuterRadius * 1.2)) { return kZero; }
    
    // Define an arc that cyclically folds and unfolds
    float phiStart = mix(mod(float(iFrame) * (1.0 / float(kLoadingCycle)), 1.0), 0.0, float((iFrame / kLoadingCycle) & 1));
    float phiEnd = mix(1.0, mod(float(iFrame) * (1.0 / float(kLoadingCycle)), 1.0), float((iFrame / kLoadingCycle) & 1));
    vec2 phiStartUv = vec2(cos(phiStart * kTwoPi), sin(phiStart * kTwoPi) ) * kLoadingMidpoint;
    vec2 phiEndUv = vec2(cos(phiEnd * kTwoPi), sin(phiEnd * kTwoPi)) * kLoadingMidpoint;
    
    // Sub-sample this function so we get nice anti-aliasing
    vec3 sum = kZero;
    for(int y = 0; y < kLoadingAASamples; y++)
    {
        for(int x = 0; x < kLoadingAASamples; x++)
        {
            // Derive the uv coordinates in the range [-0.5, 0.5], plus the radius from the center
            vec2 uv = xyToUv(xy * float(kLoadingAASamples) + vec2(x, y), iResolution * float(kLoadingAASamples)) - vec2(0.5);   
            float r = length(uv);
            
            // If we're inside the outer band, draw it and we're done.
            if(r > kLoadingOuterRadius * 1.1 && r < kLoadingOuterRadius * 1.14) { sum += vec3(0.2); continue; }
            
            // If we're outside the perimiter of the main arc, we're done.
            if(r < kLoadingInnerRadius || r > kLoadingOuterRadius) { continue; }

            // The angle of phi, normalised to [0, 1]
            float phi = (atan(-uv.y, -uv.x) + kPi) / kTwoPi;            
            
            // If we're outside the arc...
            if(phi < phiStart || phi > phiEnd)
            {
               // If we're within reach of an endpoint, set phi to that point. Otherwise, we're done.
               if(length2(phiStartUv - uv) < kLoadingThicknessSqr) { phi = phiStart; }
               else if(length2(phiEndUv - uv) < kLoadingThicknessSqr) { phi = phiEnd; }
               else { continue; }
            }           
            
            sum += vec3(0.8);
        }
    }
    
    return sum / float(kLoadingAASamples * kLoadingAASamples);
}