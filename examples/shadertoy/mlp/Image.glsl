#define kCaptureMode 0

#define kSpectralIntegrator true
#define kSpectralIntegratorMode 1
#define kStratifyLambda true
#define kDielectricDispersion 50.0

#define kMBlurGain      40.0
#define kMaxIterations  4
#define kZoomOrder      3
#define kEndPause       0.0
#define kSpeed          0.3

#if kCaptureMode == 1
    #define kCaptureTimeDelay 10.0
    #define kCaptureTimeSpeed 0.25
#else
    #define kCaptureTimeDelay 0.0
    #define kCaptureTimeSpeed 1.0
#endif

#if kClipView == 0
    #define kVignetteStrength         0.8             // The strength of the vignette effect
    #define kVignetteScale            0.7            // The scale of the vignette effect
    #define kVignetteExponent         2.0             // The rate of attenuation of the vignette effect
#else
    #define kVignetteStrength         0.8            
    #define kVignetteScale            1.0            
    #define kVignetteExponent         3.0           
#endif

vec4 Tap(int idx)
{
    return texelFetch(iChannel1, ivec2(idx % int(iResolution.x), idx / int(iResolution.x)), 0);
}

vec3 ParamColourMap(float param)
{
    #define kParamColourGain (float(kMLPWidth) * 0.5)
    float sigma = 1.0 / (1.0 + exp(-param * kParamColourGain));
    return (sigma < 0.5) ? mix(vec3(1.0, 0.0, 0.0), kOne * 0.3, sigma * 2.0) : mix(kOne * 0.3, vec3(0.0, 1.0, 0.0), sigma * 2.0 - 1.0);
}

vec3 DrawEpochClock(vec2 xy, vec3 rgb)
{
    int frame = ((iFrame - 1) / kMLPFrameSkip) % kFramesPerEpoch;
    
    #define kClockRadius 20.0
    #define kClockPos vec2(iResolution.x - kClockRadius - 10.0, kClockRadius + 10.0)    
    
    float circle = SDFCircle(xy, kClockPos, kClockRadius, 1.5, true);       

    xy -= kClockPos;
    if(frame >= int(float(kFramesPerEpoch) * (atan(-xy.x, -xy.y) + kPi) / kTwoPi))
        rgb = mix(rgb, kOne, circle);
        
    return rgb;
}

vec3 DrawLoss(vec2 xy)
{     
    #define kPlotWidth             1.4
    #define kPlotHeight            0.2
    #define kPlotEpochs            1024
    
    vec2 uvPlot = (xy - vec2(iResolution.x * 0.5, 150.f * 0.5)) / float(iResolution.y) + vec2(kPlotWidth, kPlotHeight) * 0.5;
    
    if(uvPlot.x < 0.0 || uvPlot.x > kPlotWidth || uvPlot.y < 0.0 || uvPlot.y > kPlotHeight) { return kZero; }
    
    if(uvPlot.x < gDxyDuv || uvPlot.y < gDxyDuv) { return vec3(1.0, 0.0, 0.0); }
    
    int currentEpoch = ((iFrame - 1) / kMLPFrameSkip) / kFramesPerEpoch;
    int epochIdx = int(float(kPlotEpochs) * uvPlot.x / kPlotWidth);
    
    if(epochIdx + 2 > currentEpoch) { return kZero; }

    vec4 LMax = texelFetch(iChannel1, ivec2(0, int(iResolution.y) - 1), 0) * 1.1;
    vec4 L0 = texelFetch(iChannel1, ivec2(epochIdx, int(iResolution.y) - 1), 0) / LMax;
    vec4 L1 = texelFetch(iChannel1, ivec2(epochIdx+1, int(iResolution.y) - 1), 0) / LMax;
    
    float line = SDFLine(uvPlot, 
                        vec2(kPlotWidth * float(epochIdx) / float(kPlotEpochs), kPlotHeight * L0.x), 
                        vec2(kPlotWidth * float(epochIdx + 1) / float(kPlotEpochs), kPlotHeight * L1.x), 
                        0.002);
                        
    return kOne * line;
}

vec3 DrawNetwork(vec2 xy)
{
    #define kNetworkWidth             1.4
    #define kNetworkHeight            0.8    
    #define kNetworkNodeRadius        0.015
    
    #define kModeBias                 0
    #define kModeActivation           1
    #define kModeError                2
    #define kNetworkNodeMode          kModeActivation
    
    #define kNetworkNodeDiam          (2.0 * kNetworkNodeRadius)
    #define kNetworkWeightsWidth      ((kNetworkWidth - float(kMLPNumLayers + 1) * kNetworkNodeDiam) / float(kMLPNumLayers))
    #define kNetworkNodePadding       ((kNetworkHeight - float(kMLPWidth) * kNetworkNodeDiam) / float(kMLPWidth - 1))
    #define kAntiAlias                (1.0 / min(iResolution.x, iResolution.y))
    

    #define kNetworkLayerWidth       (kNetworkNodeDiam + kNetworkWeightsWidth)
    
    vec2 uvNetwork = TransformScreenToWorld(xy) + vec2(kNetworkWidth, kNetworkHeight) * 0.5;
    
    if(uvNetwork.x < 0.0 || uvNetwork.x > kNetworkWidth || uvNetwork.y < 0.0 || uvNetwork.y > kNetworkHeight) { return kZero; }
    
    vec3 rgb = kZero;
    vec2 uvLayer = vec2(mod(uvNetwork.x - kNetworkNodeRadius, kNetworkLayerWidth), uvNetwork.y);    
    int layerIdx = int((uvNetwork.x - kNetworkNodeRadius + kNetworkLayerWidth) / kNetworkLayerWidth) - 1;  

    // Render the weights
    if(layerIdx >= 0 && layerIdx < kMLPNumLayers)
    {    
        for(int n = 0; n < kMLPWidth; ++n)
        {
            vec2 vn = vec2(kNetworkLayerWidth, kNetworkHeight * (1.0 - (float(n) / float(kMLPWidth) + kNetworkNodeRadius)));
            for(int m = 0; m < kMLPWidth; ++m)
            {
                vec2 vm = vec2(vn.x - kNetworkLayerWidth, kNetworkHeight * (1.0 - (float(m) / float(kMLPWidth) + kNetworkNodeRadius)));
                float line = SDFLine(uvLayer, vn, vm, 0.002);
                
                if(line > 0.0)
                {
                    vec4 params = Tap(kMLPLayerStride * layerIdx + kMLPRowStride * (n / 2) + (m / 2));                  
                    rgb = mix(rgb, ParamColourMap(params[2 * (n % 2) + (m % 2)]), line);
                }
            }
        }
    }    

    // Render the biases, activations and errors
    layerIdx = int((uvNetwork.x - kNetworkNodeDiam + kNetworkLayerWidth) / kNetworkLayerWidth) - 1;   
    uvLayer = vec2(mod(uvNetwork.x - kNetworkNodeDiam, kNetworkLayerWidth), uvNetwork.y);    
    if(uvLayer.x > kNetworkLayerWidth - kNetworkNodeDiam * 3.5)
    {
        for(int n = 0; n < kMLPWidth; ++n)
        {
            //for(int type = 0; type < 3; ++type)
            {            
                vec2 c = vec2(kNetworkLayerWidth - kNetworkNodeRadius,
                              kNetworkHeight - (kNetworkHeight * float(n) / float(kMLPWidth) + kNetworkNodeRadius));
                
                float circle = SDFCircle(uvLayer, c, kNetworkNodeRadius , kAntiAlias, true);            
                
                if(circle > 0.0) 
                {
                    if(layerIdx >= 0)
                    {                                    
                        vec4 params = Tap(kMLPLayerStride * layerIdx + kMLPBiasOffset + (kMLPActStride * kNetworkNodeMode) + n / 4);
                        rgb = mix(rgb, ParamColourMap(params[n % 4]), circle);
                    }
                    else
                    {
                        rgb = mix(rgb, kOne * 0.5, circle);
                    }
                    break; 
                }
            }
        }
    }    
    
    return rgb;
}

void mainImage( out vec4 rgba, in vec2 xy )
{    
    SetGlobals(xy, iResolution.xy);    
    PCGInitialise(HashOf(uint(iFrame)));
    //gDxyDuv = 0.002;
    
    if(!IsTexelInClipRegion(xy)) { return; }   
     
    vec3 rgb = DrawNetwork(xy);
    
    rgb += DrawLoss(xy);    
        
    rgb = DrawEpochClock(xy, rgb);
    
    xy /= vec2(4.0, 50.0);
    int texelIdx = int(xy.y) * int(iResolution.x) + int(xy.x) / 4;    
    if(texelIdx < kMLPNetStride)    
    {
        // The index of the layer associated with this texel
        int layerIdx = (texelIdx % kMLPNetStride) / kMLPLayerStride;    
        // The index of the parameter in the current layer
        int paramIdx = (texelIdx % kMLPNetStride) % kMLPLayerStride;
        
        if(IsError(paramIdx))        
            rgb += ParamColourMap(texelFetch(iChannel1, ivec2(int(xy.x) / 4, xy.y) / kScreenDownsample, 0)[int(xy.x) % 4]);
    } 

    rgba.xyz = rgb;    
    rgba.w = 1.0;
    
    //rgba = texelFetch(iChannel1, ivec2(xy), 0);
}