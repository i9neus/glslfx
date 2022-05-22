bool RayBox(inout Ray ray, out HitCtx hit, in Transform transform)
{
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
   
    vec3 tNearPlane, tFarPlane;
    for(int dim = 0; dim < 3; dim++)
    {
        if(abs(localRay.d[dim]) > 1e-10)
        {
            float t0 = (0.5 - localRay.o[dim]) / localRay.d[dim];
            float t1 = (-0.5 - localRay.o[dim]) / localRay.d[dim];
            if(t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
            else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
        }
    }    
   
    float tNearMax = cwiseMax(tNearPlane);
    float tFarMin = cwiseMin(tFarPlane);
    if(tNearMax > tFarMin) { return false; }  // Ray didn't hit the box
    
    float tNear;
    if(tNearMax > 0.0) { tNear = tNearMax; }
    else if(tFarMin > 0.0) { tNear = tFarMin; }
    else { return false; } // Box is behind the ray
    
    if(tNear > ray.tNear) { return false; }

    vec3 hitLocal = localRay.o + localRay.d * tNear;
    int normPlane = (abs(hitLocal.x) > abs(hitLocal.y)) ? 
                    ((abs(hitLocal.x) > abs(hitLocal.z)) ? 0 : 2) : 
                    ((abs(hitLocal.y) > abs(hitLocal.z)) ? 1 : 2);    
    vec3 n = kZero;
    n[normPlane] = sign(hitLocal[normPlane]);        
    vec3 nLocal = n + hitLocal;
    
    ray.tNear = max(0.0, tNear);
    hit.n = transpose(transform.rot) * n * transform.sca;
    if(dot(n, localRay.d) > 0.0) { hit.n = -hit.n; }
    hit.kickoff = 1e-5;
    SetRayFlag(ray, kFlagsBackfacing, cwiseMax(abs(localRay.o)) < 0.5);  
    
    return true;
}