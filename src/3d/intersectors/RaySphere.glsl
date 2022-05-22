// Ray-sphere intersection test
bool RaySphere(inout Ray ray, out HitCtx hit, in Transform transform)
{
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    
    // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
    float a = dot(localRay.d, localRay.d);
    float b = 2.0 * dot(localRay.d, localRay.o);
    float c = dot(localRay.o, localRay.o) - 1.0;
    
    float t0, t1;
    if(!SolveQuadratic(a, b, c, t0, t1)) { return false; }
    
    if(t1 < t0) 
    { 
        float swap = t1;
        t1 = t0;
        t0 = swap; 
    }

    vec3 n;
    float tNear = ray.tNear;
    if(t0 > 0.0 && t0 < tNear)
    {
        n = localRay.o + localRay.d * t0;
        tNear = t0;
    }
    else if(t1 > 0.0 && t1 < tNear)
    {
        n = localRay.o + localRay.d * t1;
        tNear = t1;
    }
    else { return false; }
    
    ray.tNear = tNear;
    hit.n = transpose(transform.rot) * n * transform.sca;
    hit.kickoff = 1e-5;
    SetRayFlag(ray, kFlagsBackfacing, dot(localRay.o, localRay.o) < 1.0);
    
    return true;
}