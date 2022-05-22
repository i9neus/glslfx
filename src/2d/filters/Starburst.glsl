vec3 Starburst(vec2 fragCoord, vec3 iResolution, ivec2 delta, vec2 bloomDimensions, float starShape, vec3 burnIn, sampler2D renderSampler)
{        
    #define kStarDownsample          3              // How much the buffer is downsampled. Higher value = lower quality, but faster 
    #define kStarWeight              vec3(0.6, 0.8, 1.0) // The per-channel weight coefficient    

    vec2 scaledResolution = vec2(iResolution.x, iResolution.y) / float((delta.x == 1) ? 1 : 1);
   
    if(fragCoord.x > scaledResolution.x || fragCoord.y > scaledResolution.y) { return kZero; }
    
    float bloomSize = (delta.x == 1) ? bloomDimensions.x : bloomDimensions.y;
    float bloomTheta = 0.5 * kHalfPi;
    mat2 rot = mat2(sin(bloomTheta), cos(bloomTheta), -cos(bloomTheta), sin(bloomTheta));
    
    int kKernelWidth = int(bloomSize * max(iResolution.x, iResolution.y) + 0.5) / ((delta.x == 1) ? 1 : 1);
    vec3 sumWeights = vec3(0.0);
    vec3 sumRgb = vec3(0.0);
    for(int i = -kKernelWidth; i <= kKernelWidth; i++)
    {      
        //vec2 xy = vec2(fragCoord.x + float(i * delta.x), fragCoord.y + float(i * delta.y));
        
        vec2 xy = (vec2(i * delta.x, i * delta.y) * rot) + vec2(fragCoord);
        
        //if(delta.x == 1) { xy *= float(1); }
        //else { xy /= float(1); }
        
        if(xy.x < 0.0 || xy.x > iResolution.x || xy.y < 0.0 || xy.y > iResolution.y) { continue; }
            
        vec4 texel = texture(renderSampler, xy / iResolution.xy);
        vec3 rgb = max(texel.xyz / max(1.0, texel.w), vec3(0.0));            
        float d = float(abs(i)) / float(kKernelWidth);
           
        vec3 weight = kOne;
        if(i != 0)
        {
            // Currently using a single weight although this effect can be done per-channel
            weight = pow(max(vec3(0.0), (exp(-sqr(d * kStarWeight * 4.0)) - 0.0183156) / 0.981684), vec3(starShape));
            //float kernel = 1.0 - pow(d, starShape);
        }
            
        sumRgb += max(kZero, rgb - burnIn) * weight;         
        sumWeights += weight;
    }
    
    sumRgb = sumRgb / sumWeights;
    
    return sumRgb;
}