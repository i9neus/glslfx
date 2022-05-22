// Ray-sphere intersection test
bool RayMetaballs(inout Ray ray, out HitCtx hit, in Transform transform, in Timecode time, inout vec3 L)
{
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    float localMag = length(localRay.d);
    localRay.d /= localMag;    
    
    #define kMetaNumIsos 20
    #define kMetaIsosurface 1.0
    #define kMetaMaxIters 40
    #define kMetaCutoffThreshold 1e-5
    #define kMetaNewtonStep 0.8
    #define kChargeGain 0.02
    
    vec4 M[kMetaNumIsos];    
    
    vec3 p;
    int iterIdx;
    float node;
    bool isSubsurface;
    float tMax = 0.0, tMin = kFltMax;
    float dotod = dot(localRay.o, localRay.d);
    vec3 dFdn = kZero;
    
    // Determine the min/max value of t along the ray that represents a valid intersection
    uvec4 pushSeed = rngSeed;
    PCGInitialise(878625u);
    for(int idx = 0; idx < kMetaNumIsos; ++idx)
    {             
       // Initialise the metaball positions
       vec4 xi = Rand();
       float theta = kTwoPi * mix(-1.0, 1.0, xi.x) + time.x * xi.y;
       float phi = kTwoPi * mix(-1.0, 1.0, xi.z) + time.y * xi.w;
       
       xi = Rand();
       vec3 p = vec3(cos(theta) * mix(0.1, 1.0, xi.z), 
                     sin(phi) * mix(0.1, 1.0, xi.z), 
                     sin(phi) * mix(0.1, 1.0, xi.w));      
       
       M[idx].xyz = p;
       M[idx].w = mix(0.1, 0.2, pow(xi.z, 2.5)) * kChargeGain;
       
       float tPerp = dot(M[idx].xyz, localRay.d) - dotod;
       tMax = max(tMax, tPerp);
       tMin = min(tMin, tPerp - 5.0 * sqrt(M[idx].w));
    }
    tMin = max(0., tMin);
    rngSeed = pushSeed;
    
    // March along the ray
    float t = tMin;
    float FMax = -kMetaIsosurface;
    float c, r, chi;    
    for(iterIdx = 0; iterIdx < kMetaMaxIters; ++iterIdx)
    {
        p = localRay.o + localRay.d * t;
        
        #define freq 30.
        #define mag 0.05
        //p += vec3(sin(p.x * freq) * mag, sin(p.y * freq) * mag, sin(p.z * freq) * mag); 
        
        //vec3 np = normalize(p);
        //p *= 1.0 + 0.05 * sin(5.0 * acos(np.y)) * cos(5.0 * acos(np.x));
        
        if(iterIdx > 0) p += normalize(dFdn) * mag;
        
        //if(iterIdx > 0) p += normalize(dFdn) * 0.2 / clamp(length(dFdn) + 1.0, 0.0, 5.0), 
        
        //if(iterIdx > 0) p += normalize(dFdn) * 0.2 / clamp(length(dFdn) + 1.0, 0.0, 5.0), 
        
        c = 0.0, r = 0.0;
        float sumw = 0.0;
        float F = 0.;
        dFdn = kZero;
        chi = 0.0;
		for(int idx = 0; idx < kMetaNumIsos; ++idx)
        {
           vec3 m;
           float cj;
           
           // Torus
           vec3 vj = M[idx].xyz, dv = M[(idx+1)%kMetaNumIsos].xyz - vj;
           vec3 vk = p + ((dot(vj, dv) - dot(p, dv)) / dot(dv, dv)) * dv - vj;
           m = vj + normalize(vk) * 0.3;           
           
           // Rods
           vj = M[idx].xyz, dv = M[(idx+1)%kMetaNumIsos].xyz - vj;
           m = mix(m, vj + saturate((dot(p, dv) - dot(vj, dv)) / dot(dv, dv)) * dv, time.y);
           cj = M[idx].w;           
           
           // Balls
           //m = M[idx].xyz;
           //cj = M[idx].w;
          
           /*vec3 vj = M[idx].xyz, dv = M[(idx+1)%kMetaNumIsos].xyz - vj;
           float tj = saturate((dot(p, dv) - dot(vj, dv)) / dot(dv, dv));
           float cj = M[idx].w;
           vec3 m = vj + tj * dv; */
           //vec3 m = M[idx].xyz;
           
           float ri = length(p - m);
           float w = 1.0 / max(1e-10, ri*ri);
           F += cj * w;
           
           // Compute the gradient of the field
           dFdn += 2.0 * cj * (p - m) * sqr(w);
           
           r += 1.0 / max(1e-10, ri);
           sumw += w;
           chi += w * float(idx) / float(kMetaNumIsos);
        }        
   
        r /= sumw;
        chi /= sumw;
        c = F * r*r;        
        F -= kMetaIsosurface;
        
        if(iterIdx == 0) { isSubsurface = (F > 0.0); }
        
        if (abs(F) < kMetaCutoffThreshold * ((F > 0.0) ? 1.0 : length(dFdn))) { break; }

		float dt = (-sqrt(c) + r * kMetaIsosurface) / kMetaIsosurface;        
        dt *= -sign(F) * kMetaNewtonStep;
        
        if((isSubsurface && F <= 0.0) || (!isSubsurface && F > 0.0)) { dt *= -1.0; }
        
        t += dt;// * mix(0.5, 1.0, Rand().x);
        
        if(t < 0.0 || t > ray.tNear * localMag || (!isSubsurface && t > tMax)) { return false; }
    }
    t /= localMag;
    
    // Errors can cause the shading normal to point away from the ray origin. Correct for this here.
    float cosTheta = dot(dFdn, localRay.d);
    if(cosTheta > 0.0) 
    { 
        //return false;
        dFdn = dFdn - cosTheta * localRay.d;
    }
    
    ray.tNear = t;
    hit.n = normalize(transpose(transform.rot) * dFdn);
    hit.kickoff = 1e-3;
    hit.chi = chi;
    SetRayFlag(ray, kFlagsBackfacing, isSubsurface);
    
    //L = hit.n;
    //return false;    
    
    return true;
}