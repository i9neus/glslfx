int Trace(inout Ray ray, inout HitCtx hit, inout Transform transform, Timecode time, in int depth, inout vec3 L)
{ 
    transform = IdentityTransform();
    
    hit.matID = kInvalidHit;
    hit.n = kZero;
    hit.chi = -1.0;
    ray.tNear = kFltMax;
    
    if(RayCornell(ray, hit, transform)) { hit.matID = kMatCornellBox; }
    
    //transform = CompoundTransform(vec3(-0., 0.1, 0.0), vec3(0.0, kTwoPi * 0.9, kHalfPi), 0.2);
    //if(RaySimplexSDF(ray, hit, transform, 0.4)) { hit.matID = kMatDielectric; }   
    //if(RayBox(ray, hit, transform)) { hit.matID = kMatDielectric; }   
    
    transform = CompoundTransform(vec3(-0.0, -0.1, 0.0), vec3(0.0, 0.0, 0.0), 0.4);
    if(RayMetaballs(ray, hit, transform, time, L)) { hit.matID = kMatDielectric; }  
    
    #define kEmitterSize 0.5
    #define kEmitterRadiance (1.0 * kOne / sqr(kEmitterSize))
    //#define kEmitterRadiance (1. * kOne)

    transform = CompoundTransform(vec3(0., 0.48, 0.0), vec3(kHalfPi, 0.0, toRad(0.)), kEmitterSize);
    if(RayPlane(ray, true, hit, transform)) { hit.matID = kMatLight; }
    
    return hit.matID;
}

bool Shade(inout Ray ray, inout HitCtx hit, inout Transform lightTransform, inout vec3 L, Timecode time, int depth, int sampleIdx, int mlpMode, ivec2 resolution, sampler2D mlpSampler)
{     
    // If this sample is a light ray, all we need to know is whether or not it hit the light. 
    // If it did, just accumulate the weight which contains the radiant energy from the light sample. 
    if(IsLightSample(ray))
    {
        if(hit.matID == kMatLight && !IsBackfacing(ray) && (mlpMode != kMLPTrain || IsCausticPath(ray)))
        {
            #ifndef kDisableLights
                //if(mlpMode == kMLPTrain)
                    L += ray.weight;
            #endif
        }
        return false;
    } 

    //if(hit.matID == kMatIncandescent) L += ray.weight * kOne;
  
    // Generate some random numbers
    vec4 xi = Rand();
    
    #define kSampleDirectEmitter  1
    #define kSampleDirectBxDF     2
    #define kSampleGuidedHG       4
    #define kSampleGuidedIso      8
    #define kSampleIndirect       16

    // Construct a cumulative mass function for the 4 sample schemes
    vec4 cmf;  
    switch(mlpMode)
    {
    case kMLPTrain: cmf = vec4(0.0, 0.0, 0.0, 0.0); break;
    case kMLPEvaluate: 
        cmf = vec4(.25, 0.5, 0.75, 1.0); 
        break;
    case kMLPReference: cmf = vec4(0.25, 0.5, 0.5, 0.5); break;
    default: return false;
    }

    // If the surfaace is specular dielectric, only sample unguided indirect paths 
    if(hit.matID == kMatDielectric) { cmf = vec4(0.0, 0.0, 0.0, 0.0); }    
    
    // Handle the light as a perfect blackbody that reflects no energy
    if(hit.matID == kMatLight)
    {
        if(!IsBackfacing(ray) && (IsCausticPath(ray) || (mlpMode != kMLPTrain && !IsScattered(ray))))
        {
            #ifndef kDisableLights
                //if(mlpMode == kMLPTrain)
                    L += kEmitterRadiance * ray.weight;
            #endif
        }
        return false;
    }  
    
    // Decide which sample scheme we're going to draw from
    int sampleOp = (xi.x < cmf[1]) ? ((xi.x < cmf[0]) ? kSampleDirectEmitter : kSampleDirectBxDF) :
                                     ((xi.x < cmf[3]) ? ((xi.x < cmf[2]) ? kSampleGuidedHG : kSampleGuidedIso) : 
                                                        kSampleIndirect);
                                                        
    // Disable multiple bounce volumetric illumination
    if(IsScattered(ray) && !IsPerfectSpecularBxDF(hit)) 
    { 
        cmf = vec4(0.5, 1.0, 1.0, 1.0); 
        ray.weight *= 0.5;
    }
                                     
    ///////////////// MLP /////////////////
    
    float pdfLight = 0.0, pdfBxDF = 0.0, pdfGuided = 0.0;
    float bxdfWeight = 1.0;
    vec4 axis;
    float g; 
    
    
    ///////////////// SAMPLE /////////////////
    
    // Sample the BxDF
    if((sampleOp & (kSampleIndirect | kSampleDirectBxDF | kSampleIndirect)) != 0) 
    {
        switch(hit.matID)
        {
        case kMatDielectric: 
            pdfBxDF = SampleSpecularDielectric(ray, hit); 
            //pdfBxDF = SampleGGXDielectric(ray, hit);
            break;
        case kMatCornellBox:
            pdfBxDF = SampleLambertianBDRF(ray, hit, sampleIdx, depth, L); 
            break;     
        default:
            return false;
        }
        
        if(pdfBxDF <= 0.) { return false; }
    }   
    
    // Sample the light
    if(sampleOp == kSampleDirectEmitter)
    {                            
        pdfLight = SampleQuadLight(ray, hit, lightTransform, kEmitterRadiance, sampleIdx, depth, L);         
        if(pdfLight <= 0.) { return false; }
    }      
    
    ///////////////// EVALUATE /////////////////
    
    // Evaluate the BxDF
    if((sampleOp & (kSampleDirectEmitter | kSampleGuidedHG | kSampleGuidedIso | kSampleIndirect)) != 0)
    {        
        switch(hit.matID)
        {
        case kMatDielectric: 
            pdfBxDF = EvaluateSpecularDielectric(); break;
            //pdfBxDF = EvaluateGGXDielectric(); break;
            break;
        case kMatCornellBox:
            pdfBxDF = EvaluateLambertianBRDF(ray.od.d, hit.n, bxdfWeight); break;  
        default:
            return false;
        }        
    }  
    
    // Evaluate the light
    if(sampleOp == kSampleDirectBxDF)
    {
        pdfLight = EvaluateQuadLight(ray, hit, lightTransform, kEmitterRadiance);
    }
    
    //if(sampleOp == kSampleDirectEmitter || sampleOp == kSampleDirectBxDF) return false;    
    
    ///////////////// COMBINE ////////////////
    
    // If we're doing MIS, reweight now
    if(sampleOp == kSampleDirectEmitter)
    { 
        //ray.weight = kZero;
        ray.weight *= 2.0 * bxdfWeight * PowerHeuristic(pdfLight, pdfBxDF);
    }
    else if(sampleOp == kSampleDirectBxDF)
    { 
        //ray.weight = cwiseMax(ray.weight) * kOne;
        ray.weight *= 2.0 * bxdfWeight * PowerHeuristic(pdfBxDF, pdfLight);
    }
    else if(sampleOp == kSampleGuidedHG)
    {
        //ray.weight = kZero;
        ray.weight *= 2.0 * bxdfWeight * PowerHeuristic(pdfGuided, 1.0 / (IsVolumetricBxDF(hit) ? kFourPi : kTwoPi)); 
    }
    else if(sampleOp == kSampleGuidedIso)
    {        
        //ray.weight = kZero;
        ray.weight *= 2.0 * bxdfWeight * PowerHeuristic(1.0 / (IsVolumetricBxDF(hit) ? kFourPi : kTwoPi), pdfGuided); 
    }    
    else if(sampleOp == kSampleIndirect)
    {
        //ray.weight = kZero;
        ray.weight *= bxdfWeight;
    }
    
    // Weight the ray depending on whether duel sampling schemes are in use or not
    if(hit.matID != kMatDielectric && cmf[0] != 0.0) { ray.weight *= 2.0; }
    
    ///////////////// FINISH UP ///////////////// 
                
    //if(depth == 1 && IsScattered(ray)) L = ray.weight * kRed;
    //L += kGreen;
    
    // Direct sampling so mark the ray as a light sample
    if((sampleOp & (kSampleDirectEmitter | kSampleDirectBxDF)) != 0)
    {
        ray.flags |= kFlagsLightSample; 
    }       
    
    // Get the surface albedo
    vec3 albedo = kOne * 0.7;
    if(hit.matID == kMatCornellBox || hit.matID == kMatVolume) 
        ray.weight *= albedo;   
        
    if(hit.matID == kMatDielectric) ray.weight *= Hue(hit.chi * 2.0);

    // Is the extant ray travelling below the subsurface?
    SetRayFlag(ray, kFlagsSubsurface, dot(ray.od.d, hit.n) < 0.0); 
    
    //if(depth > 1 && !IsCausticPath(ray) && sampleOp == kSampleDirectEmitter) return false;
    //if(depth > 1 && !IsCausticPath(ray) && sampleOp == kSampleDirectBxDF) return false;
    //if(sampleOp == kSampleIndirectBxDF) return false;
    //if(sampleOp == kSampleIndirectMLP) return false;
    
    return true;
}

bool IntegratePath(inout Ray ray, inout vec3 L, in Timecode time, int sampleIdx, int mlpMode, ivec2 resolution, sampler2D mlpSampler)
{
    // Interatively sample the path
    HitCtx hit; 
    Transform transform;
    
    #define kMaxDepth        4
    for(int depth = 0; depth < kMaxDepth; depth++)
    {
        if(Trace(ray, hit, transform, time, depth + 1, L) == kInvalidHit) { break; }    
        
        //L = hit.n;
        //return false;
        
        if(!Shade(ray, hit, transform, L, time, depth + 1, sampleIdx, mlpMode, resolution, mlpSampler)) { break; }
    }   
    
    // Splat clamp
    //L.xyz = min(vec3(1e5), L.xyz);
    
    //if(any(isnan(L))) L = kRed;
     
    return true;
}
