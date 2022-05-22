#define kCaptureMode 0

#define kSpectralIntegrator false
#define kSpectralIntegratorMode 1
#define kLambdaSampler 0

#define kDielectricDispersion 10.0

#define kMBlurGain      .0
#define kMaxIterations  4
#define kZoomOrder      3
#define kEndPause       0.0
#define kSpeed         2.5
#define kMaxDepth       2

#define kBackgroundColour 0.
#define kEdgeGain 1.0 
#define kEdgeEmphasisGain 0.0
#define kFaceEmphasisGain 0.0

#define kOriginCellHeadStart 0.1
#define kCubeShrink 0.6
#define kCubeScaleEaseIn 0.05
#define kCubeScaleEaseOut 0.25

#if kCaptureMode == 1
    #define kCaptureTimeDelay 20.0
    #define kCaptureTimeSpeed (2.0 / 60.0)
#else
    #define kCaptureTimeDelay 0.0
    #define kCaptureTimeSpeed 1.0
#endif

#if kClipView == 0
    #define kSternographicScale 1.1
    #define kViewScale 1.
#else
    #define kSternographicScale 1.3
    #define kViewScale 0.25
#endif


float CauchyPDF(float x, float D)
{
    return 1.0 / (kPi * D * (1.0 + sqr(x / D)));
}

float CauchyCDF(float x, float D)
{
    return kInvPi * atan(x / D) + 0.5;
}

void EquiangularSample(in float L0, in float L1, in float D, in float xi, out float tEqui, out float pdf)
{
    float cdf0 = CauchyCDF(L0, D);
    float cdf1 = CauchyCDF(L1, D);
    tEqui = tan((mix(cdf0, cdf1, xi) - 0.5) * kPi) * D;
        
    pdf = CauchyPDF((tEqui - L0) / (L1 - L0), D) / max(1e-10, cdf1 - cdf0);
}

bool TraceWavefront(inout Ray ray, out vec4 L, vec3 time, int depth, int sampleIdx, int maxSamples, bool isVolumePass)
{      
    float phi = KickDrop(time.y, vec2(0.0, 0.0), vec2(0.5, 1.0), vec2(kClosedInterval - 3, 0.7), vec2(kClosedInterval, 0.0));
    float alpha = KickDrop(time.y, vec2(0.0, 0.0), vec2(0.5, 1.0), 
                                   vec2(float(kClosedInterval), 1.0), vec2(float(kClosedInterval) + 0.5, 0.0));
    float beta = KickDrop(time.z, vec2(0.0, 0.0), vec2(0.4, 1.2), vec2(0.5, 1.0));
    float theta = 2.0 * kTwoPi * time.x / float(kNumIntervals);
    float flare = KickDrop(time.y, vec2(0.0, 2.0), vec2(0.2, 3.0), vec2(0.8, 1.0));
    
    int matID = -1;
    vec2 sdfNear = vec2(kFltMax);
    Transform transform;
    HitCtx hit;
    ray.tNear = kFltMax;
    
    transform = CompoundTransform(vec3(0.0), vec3(0.0, theta, 0.5 * theta), 1.1);
    if(RaySimplexSDF(ray, sdfNear, hit, transform, mix(0.2, 1.5, phi), int(time.y), time.z)) { matID = 0; }          
        
        #define kExtinction 0.1
        if((ray.flags & kFlagsDirectRay) != 0u)
        {
            float distToLight = length2(ray.od.o);
            if(matID != -1 && sqr(ray.tNear) < distToLight) { return false; } 
            
            L.xyz += ray.weight; return false;

            if(sqr(sdfNear.y) > distToLight)
                L.xyz += ray.weight;
            else
                L.xyz += ray.weight * min(1.0, 2.0 * sqrt(abs(sdfNear.x) / sdfNear.y));

            return false;
        }

        
        float xi = fract((float(sampleIdx) + Rand(iChannel0).x) / float(maxSamples));
        float tPerp = max(0.0, -dot(ray.od.o, ray.od.d));
        float tVolume, pdfCauchy;
        float tDepth = (matID == -1) ? 5.0 : ray.tNear;
        float D = max(0.1, length(RayPointAt(ray, tPerp)));
        
        EquiangularSample(-tPerp, tDepth - tPerp, D, xi, tVolume, pdfCauchy);        
        tVolume += tPerp;
        
        uint packed = uint(float(0xffff) * min(1.0, ray.tNear / 5.0)) |
                      uint(float(0xffff) * (tVolume / 5.0)) << 16;
                      
        L.w = uintBitsToFloat(packed);
        
        if(tVolume > ray.tNear) { return false; }

        ray.od.o = ray.od.o + ray.od.d * tVolume;
        float distToLight = length(ray.od.o);
        ray.od.d = -ray.od.o / distToLight;
        ray.flags = kFlagsDirectRay;
        ray.weight *= 3.0 * (1.0 / max(1.0, sqr(distToLight))) * exp(-tVolume * kExtinction) * kExtinction / pdfCauchy;
        
        float arc = 0.4 * (theta * 3.0 + atan(ray.od.d.y, ray.od.d.z) + acos((ray.od.d.x > 0.0) ? sqr(ray.od.d.x) : (-sqr(-ray.od.d.x))));
        ray.weight = pow(ray.weight * kOne, vec3(0.8)) * flare;
        
        return true;    
}

vec4 TracePath(vec2 uvScreen, int sampleIdx, int maxSamples, bool isVolumePass)
{
    // Sample the spectrum
    float xi = (float(sampleIdx) + Rand(iChannel0).x) / float(maxSamples);    
    float lambda = mix(3800.0, 7000.0, xi);
    
    // Configure the time
    vec3 time;
    time.x = max(0.0, iTime - kCaptureTimeDelay);    
    float timeSubSample = 1.0 / 60.0 * xi * kMBlurGain;
    time.x = (time.x * kCaptureTimeSpeed + timeSubSample) * kSpeed;// + 3.0 * float(kNumIntervals) * 0.99;
    time.y = mod(time.x, float(kNumIntervals));
    time.z = fract(time.x);  
    
    vec2 uvView = TransformScreenToWorld(uvScreen);
    
    if(!InverseSternograph(uvView, kSternographicScale)) { return vec4(0.0); } 
    
    // Create a new camera ray and store the sampled wavelength (we reuse the same ray at each bounce)   
    Ray cameraRay = CreateOrthographicCameraRay(uvView, vec2(1.7320508) * 1.1, vec3(2.0), vec3(0.0));   
    
    // Interatively sample the path
    vec4 L = vec4(0.0);
    for(int depth = 0; depth < kMaxDepth; depth++)
    {
        // Passing Ray structures by reference breaks the shader on Macs. This explains
        // the clunky approach to returning extant rays by value.
        if(!TraceWavefront(cameraRay, L, time, depth + 1, sampleIdx, maxSamples, isVolumePass))
        {        
           if(depth == 0) { return L; }
            break; 
        }
    }
   
    return L;
}


void mainImage( out vec4 rgba, in vec2 xy )
{        
    // Split the screen into two zones, one for the noisy input, one for the denoised output
    int zone = int(xy.x > iResolution.x * 0.5);
    xy.x = mod(xy.x, iResolution.x * 0.5);    
    
    SetGlobals(xy, iResolution.xy * vec2(0.5, 1.0));
 
    vec4 rgbi = vec4(0.0);    
        
    int filterSize = (zone == 0) ? 1 : 4;
    int numSamples = sqr(filterSize);
    for(int i = 0, sampleIdx = 0; i < filterSize; ++i)
    {
        for(int j = 0; j < filterSize; ++j, ++sampleIdx)
        {
            PCGInitialise(HashOf(uint(0), uint(xy.x) % 2u, uint(xy.y) % 2u));
            vec2 xyAA = xy + vec2(float(i) / float(filterSize), float(j) / float(filterSize));                        

            rgbi += TracePath(xyAA, (zone == 0) ? int(OrderedDither() * 16.0) : sampleIdx, 16, true);
        }
    }
    
    rgba = rgbi / float(numSamples);    
  
}