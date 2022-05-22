#define kCaptureMode 1

#define kSpectralIntegrator true
#define kSpectralIntegratorMode 1
#define kLambdaSampler 0

#define kDielectricDispersion 10.0

#define kMBlurGain      7.0
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

#define kSkylightFloor 0.
    #define kSkylightCeil 1.0
    #define kSkylightGainA 1.0
    #define kSkylightGainB 1.0
    #define kSkylightFreqA 5.0
    #define kSkylightFreqB 20.0
    #define kSkylightNormalA vec3(1.0, 0.0, 0.0)
    #define kSkylightNormalB vec3(0.0, -1.0, 0.0)
    #define kSkylightHardness 5.0
    #define kEnvironmentRadius 2.0
#define kTestSkylight false

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

vec3 Skylight(vec3 n)
{
    float thetaA = acos(dot(n, normalize(kSkylightNormalA)));
    float thetaB = acos(dot(n, normalize(kSkylightNormalB)));    
    
    float v = (kSkylightGainA * cos01(kSkylightFreqA * thetaA) + kSkylightGainB * sin01(kSkylightFreqB * thetaB)) / (kSkylightGainA + kSkylightGainB);
    
    return vec3(mix(kSkylightFloor, kSkylightCeil, pow(saturate(v * kSkylightHardness - (0.5 * (kSkylightHardness - 1.0))), 1.0)));
}

vec3 SampleEnvironmentSphere(RayBasic ray, vec3 time, float radius)
{
    float a = dot(ray.d, ray.d);
    float b = 2.0 * dot(ray.d, ray.o);
    float c = dot(ray.o, ray.o) - sqr(radius);

    float t0, t1;
    if (!QuadraticSolve(a, b, c, t0, t1)) { return kZero; }

    vec3 n = normalize(ray.o + ray.d * ((t0 > 0.0) ? t0 : t1));
    
    return Skylight(n);
}

void SamplePerfectSpecular(inout Ray ray, in HitCtx hit)
{
    vec3 r = reflect(ray.od.d, hit.n);
    
    ray.od.o = ray.od.o + ray.od.d * ray.tNear + hit.n * hit.kickoff;
    ray.od.d = r;  
}


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
        
    //tEqui = (x - L0) / (L1 - L0);
    pdf = CauchyPDF((tEqui - L0) / (L1 - L0), D) / max(1e-10, cdf1 - cdf0);
}

bool TraceWavefront(inout Ray ray, out vec4 L, vec3 time, int depth, int sampleIdx, int maxSamples, bool isVolumePass)
{   
    /*if(depth == kMaxDepth)
    {
        L.xyz += SampleEnvironmentSphere(ray.od, time) * ray.weight;
        return false;
    } */   
   
    //float phi = max(0.0, sin(kPi * (floor(time.y) + PaddedSmoothStep(time.z, 0.0, 1.0)) / float(kClosedInterval + 1)));
    float phi = KickDrop(time.y, vec2(0.0, 0.0), vec2(0.5, 1.0), vec2(kClosedInterval - 3, 0.7), vec2(kClosedInterval, 0.0));
    float alpha = KickDrop(time.y, vec2(0.0, 0.0), vec2(0.5, 1.0), 
                                   vec2(float(kClosedInterval), 1.0), vec2(float(kClosedInterval) + 0.5, 0.0));
    float beta = KickDrop(time.z, vec2(0.0, 0.0), vec2(0.4, 1.2), vec2(0.5, 1.0));
    float theta = kTwoPi * 2.0 / 3.0 * mix(time.x, floor(time.x) + beta, 0.7) / float(kNumIntervals);
    float flare = KickDrop(time.y, vec2(0.0, 2.0), vec2(0.2, 3.0), vec2(0.8, 1.0));
    
    int matID = -1;
    vec2 sdfNear = vec2(kFltMax);
    Transform transform;
    HitCtx hit;
    ray.tNear = kFltMax;
    
    transform = CompoundTransform(vec3(0.0), vec3(0.0, theta, kHalfPi), 1.1);
    if(RaySimplexSDF(ray, sdfNear, hit, transform, mix(0.2, 1.5, phi), int(time.y), time.z)) { matID = 0; }          
        
    if(!isVolumePass || depth == 1)
    {
        transform = CompoundTransform(vec3(0.0), vec3(0.0, theta, kHalfPi), .8);
        if(RaySimplexSDF(ray, sdfNear, hit, transform, 0.0, int(time.y), time.z)) { matID = 1; }
    }
   
    //L.xyz += hit.n; return false;
    
    #define kExtinction 0.1
    if(isVolumePass)
    {
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

        float xi = (float(sampleIdx) + Rand(iChannel0).x) / float(maxSamples);
        float tPerp = max(0.0, -dot(ray.od.o, ray.od.d));
        float tVolume, pdfCauchy;
        float tDepth = (matID == -1) ? 5.0 : ray.tNear;
        float D = max(0.1, length(RayPointAt(ray, tPerp)));
        
        EquiangularSample(-tPerp, tDepth - tPerp, D, xi, tVolume, pdfCauchy);        
        tVolume += tPerp;

        ray.od.o = ray.od.o + ray.od.d * tVolume;
        //ray.od.d = SampleUnitSphere(xi.yz);
        float distToLight = length(ray.od.o);
        ray.od.d = -ray.od.o / distToLight;
        ray.flags = kFlagsDirectRay;
        ray.weight *= 3.0 * (1.0 / max(1.0, sqr(distToLight))) * exp(-tVolume * kExtinction) * kExtinction / pdfCauchy;
        
        float arc = 0.4 * (theta + atan(ray.od.d.y, ray.od.d.z) + acos((ray.od.d.x > 0.0) ? sqr(ray.od.d.x) : (-sqr(-ray.od.d.x))));
        ray.weight = pow(ray.weight * Hue(fract(arc)), vec3(0.8)) * flare;
        //ray.weight *= 0.5;
        
        return true;             
    }
    
    
    if(matID == -1)
    {
        if(depth != 1){ L.xyz += SampleEnvironmentSphere(ray.od, time, mix(10.0, 2.0, phi)) * ray.weight; }
        else { L.xyz += kOne * kBackgroundColour; }
        return false; 
    }
    
    //ray.weight *= exp(-ray.tNear * kExtinction);
    
    //ray.weight *= alpha;
    /*L.xyz += (dot(-normalize(ray.od.d), hit.n)) * (1.0 - sqrt(phi)) * 0.2; 
    if(alpha == 0.0) { return false; }*/
    
    switch(matID)
    {
    case 0:
        SamplePerfectSpecular(ray, hit);
        ray.weight *= mix(0.25, 1.0, sqrt(phi));
        return true;
    case 1:  
        L.xyz += ray.weight * flare;
    }
    
    return false;
    
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
    Ray cameraRay = CreateOrthographicCameraRay(uvView, vec2(1.7320508) * 1.5, vec3(2.0), vec3(0.0));   
    //cameraRay.lambda = lambda;
    
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
      
    if(kSpectralIntegrator && !isVolumePass)
    {  
        L.xyz *= SampleSpectrum(lambda);
        //L.xyz *= Hue(xi) * 2.;
    }
   
    return L;
}


vec2 Interfere(vec2 xy)
{
    #define kStatic true
    #define kStaticFrequency 0.02
    #define kStaticLowMagnitude 0.005
    #define kStaticHighMagnitude 0.005
    
    #define kVDisplace true
    #define kVDisplaceFrequency 0.06
    
    #define kHDisplace true
    #define kHDisplaceFrequency 0.14
    #define kHDisplaceVMagnitude 0.05
    #define kHDisplaceHMagnitude 0.3
    
    float frameHash = HashToFloat(HashOf(uint(iFrame / int(10.0 / kCaptureTimeSpeed))));
    
    if(kStatic)
    {
        // Every now and then, add a ton of static
        float interP = 0.01, displacement = iResolution.x * kStaticLowMagnitude;
        if(frameHash < kStaticFrequency)
        {
            interP = 0.5;
            displacement = kStaticHighMagnitude * iResolution.x;
        }

        // CRT interference at PAL refresh rate 
        PCGInitialise(HashOf(uint(xy.y), uint(iFrame / int(60.0 / (24.0 * kCaptureTimeSpeed)))));
        vec4 xi = Rand();
        if(xi.x < interP) 
        {  
            float mag = mix(-1.0, 1.0, xi.y);        
            xy.x -= displacement * sign(mag) * sqr(abs(mag)); 
        }
    }
    
    // Vertical displacment
    if(kVDisplace && frameHash > 1.0 - kVDisplaceFrequency)
    {
        float dispX = HashToFloat(HashOf(8783u, uint(iFrame / int(10.0 / kCaptureTimeSpeed))));
        float dispY = HashToFloat(HashOf(364719u, uint(iFrame / int(12.0 / kCaptureTimeSpeed))));
        
        if(xy.y < dispX * iResolution.y) 
        { 
            xy.y -= mix(-1.0, 1.0, dispY) * iResolution.y * 0.2; 
        }
    }
    // Horizontal displacment
    else if(kHDisplace && frameHash > 1.0 - kHDisplaceFrequency - kVDisplaceFrequency)
    {
        float dispX = HashToFloat(HashOf(147251u, uint(iFrame / int(9.0 / kCaptureTimeSpeed))));
        float dispY = HashToFloat(HashOf(287512u, uint(iFrame / int(11.0 / kCaptureTimeSpeed))));
        float dispZ = HashToFloat(HashOf(8756123u, uint(iFrame / int(7.0 / kCaptureTimeSpeed))));
        
        if(xy.y > dispX * iResolution.y && xy.y < (dispX + mix(0.0, kHDisplaceVMagnitude, dispZ)) * iResolution.y) 
        { 
            xy.x -= mix(-1.0, 1.0, dispY) * iResolution.x * kHDisplaceHMagnitude; 
        }
    }
    
    return xy;
}

void mainImage( out vec4 rgba, in vec2 xy )
{    
    if(kTestSkylight)
    {
        vec3 n = vec3(cos(kTwoPi * xy.x / iResolution.x) * sin(kPi * xy.y / iResolution.y), 
                      sin(kTwoPi * xy.x / iResolution.x) * sin(kPi * xy.y / iResolution.y), 
                      cos(kPi * xy.y / iResolution.y));
        rgba.xyz = Skylight(n);
        rgba.w = 1.0;
        return;
    }
   
    if(xy.x > iResolution.x / float(kScreenDownsample) || xy.y > iResolution.y / float(kScreenDownsample)) { return; }
    
    if((xy.x - (iResolution.x * 0.5)) > iResolution.y * 0.5) { return; }
    
    SetGlobals(xy, iResolution.xy);
    
    //if(!IsTexelInClipRegion(xy)) { return; }
        
    xy *= float(kScreenDownsample);
    vec3 rgb = kZero;
    vec4 rgbi = vec4(0.0);    
    
    #define kGeometryPass
    #define kVolumePass
    #define kInterference true
    
    vec2 xyInterfere = xy;//(kInterference) ? Interfere(xy) : xy;   
    
    #ifdef kGeometryPass    
        #define kAntiAlias 4
        #define kNumSamples (kAntiAlias*kAntiAlias)
       
        for(int i = 0, idx = 0; i < kAntiAlias; ++i)
        {
            for(int j = 0; j < kAntiAlias; ++j, ++idx)
            {
                PCGInitialise(HashOf(uint(idx)));
                vec2 xyAA = xyInterfere + vec2(float(i) / float(kAntiAlias), float(j) / float(kAntiAlias));

                rgbi += TracePath(xyAA, idx, kNumSamples, false);
            }
        }    
        rgb += rgbi.xyz / float(kNumSamples);
    #endif
    
    #ifdef kVolumePass
        rgbi = vec4(0.0);
        #define kVolumeAntiAlias 3
        #define kNumVolumeSamples (kVolumeAntiAlias*kVolumeAntiAlias)
        for(int i = 0, idx = 0; i < kVolumeAntiAlias; ++i)
        {
            for(int j = 0; j < kVolumeAntiAlias; ++j, ++idx)
            {
                PCGInitialise(HashOf(uint(idx)));
                vec2 xyAA = xyInterfere + vec2(float(i) / float(kVolumeAntiAlias), float(j) / float(kVolumeAntiAlias));

                rgbi += TracePath(xyAA, idx, kNumVolumeSamples, true);
            }
        }
        rgb += rgbi.xyz / float(kNumVolumeSamples);
    #endif
    
    //rgb = mix(rgb, kOne - rgb, rgbi.w);
    
    #define kColourQuantisation 4
    /*rgb *= float(kColourQuantisation);
    if(fract(rgb.x) > OrderedDither()) rgb.x += 1.0;
    if(fract(rgb.y) > OrderedDither()) rgb.y += 1.0;
    if(fract(rgb.z) > OrderedDither()) rgb.z += 1.0;
    rgb = floor(rgb) / float(kColourQuantisation);*/
    
    // Grade
    vec3 hsv = RGBToHSV(rgb);    
    hsv.x += -sin((hsv.x + 0.0) * kTwoPi) * 0.15;
    hsv.y *= 1.0;    
    rgb = HSVToRGB(hsv);
    
    //rgb += 0.1 * float((int(xy.y) / kScreenDownsample) & 1);
    rgb += 0.05;
    
    rgba = vec4(rgb, 1.0);
}