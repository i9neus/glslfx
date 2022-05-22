// Seperable bloom function. This filter requires two passes in the horizontal and vertical directions which are combined as a post-process
// effect after each frame. The accuracy/cost of the effect can be tuned by dialing the kBloomDownsample parameter. 
vec3 Bloom(vec2 xyScreen, vec3 iResolution, ivec2 delta, vec2 bloomDimensions, float bloomShape, vec3 burnIn, sampler2D renderSampler)
{        
    // How much the bloom buffer is downsampled. Higher value = lower quality, but faster
    #define kBloomDownsample  1  

    vec2 scaledResolution = vec2(iResolution.x, iResolution.y) / float((delta.x == 1) ? kBloomDownsample : 1);
   
    if(xyScreen.x > scaledResolution.x || xyScreen.y > scaledResolution.y) { return kZero; }
    
    float bloomSize = (delta.x == 1) ? bloomDimensions.x : bloomDimensions.y;
    
    int kKernelWidth = int(bloomSize * max(iResolution.x, iResolution.y) + 0.5) / ((delta.x == 1) ? kBloomDownsample : 1);
    vec3 sumWeights = vec3(0.0);
    vec3 sumRgb = vec3(0.0);
    for(int i = -kKernelWidth; i <= kKernelWidth; i++)
    {      
        vec2 xy = vec2(xyScreen.x + float(i * delta.x), xyScreen.y + float(i * delta.y));
        
        if(delta.x == 1) { xy *= float(kBloomDownsample); }
        else { xy /= float(kBloomDownsample); }
        
        if(xy.x < 0.0 || xy.x > iResolution.x || xy.y < 0.0 || xy.y > iResolution.y) { continue; }
            
        vec4 texel = texture(renderSampler, xy / iResolution.xy);
        vec3 rgb = max(texel.xyz / max(1.0, texel.w), vec3(0.0));            
        float d = float(abs(i)) / float(kKernelWidth);
           
        vec3 weight = kOne;
        if(i != 0)
        {
            // Currently using a single weight although this effect can be done per-channel
            float kernel = (max(0.0, (exp(-pow(d * 4.0, bloomShape)) - 0.0183156) / 0.981684));            
            weight = kOne * kernel;
        }
            
        sumRgb += ((delta.y == 1) ? rgb : max(kZero, rgb - burnIn)) * weight;         
        sumWeights += weight;
    }
    
    sumRgb = sumRgb / sumWeights;
    
    return (delta.x == 1) ? sumRgb : (sumRgb * kBloomTint);
}
