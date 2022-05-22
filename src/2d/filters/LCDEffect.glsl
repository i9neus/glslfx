vec3 LCDEffect( in vec2 xyScreen, in vec3 rgb)
{
    #define kLCDAxis 0
    vec2 kLCDScale = (kLCDAxis == 0) ? vec2(3.0, 1.0) : vec2(1.0, 3.0);
    vec2 kLCDDotMask = (kLCDAxis == 0) ? vec2(1.5, 3.0) : vec2(1.5, 3.0);
        
    // Brighten and gamma adjust
    rgb = pow(rgb * 3.0, vec3(1.2));

    vec2 xyPixel = mod(xyScreen, vec2(kScreenDownsample));
    vec2 xySubpixel = mod(xyPixel, vec2(kScreenDownsample) / kLCDScale);   
    int subPixelIdx = int(xyPixel[kLCDAxis] / (float(kScreenDownsample) / 3.0));
    vec2 xyDot = ((xySubpixel - vec2(kScreenDownsample) / (2. * kLCDScale)) / (vec2(kScreenDownsample) / (2. * kLCDScale))) * vec2(1.5, 1.2);
    xyDot = sign(xyDot) * pow(abs(xyDot), kLCDDotMask);
    float dist = max(0.0, 1.0 - length(xyDot) / 2.8);
    
    rgb *= dist;    
    rgb[(subPixelIdx+1)%3] *= 0.5;
    rgb[(subPixelIdx+2)%3] *= 0.5;
    
    return rgb;
}