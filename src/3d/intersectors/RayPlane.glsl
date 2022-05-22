// Ray-plane intersection test
bool RayPlane(inout Ray ray, bool bounded, out HitCtx hit, in Transform transform)
{
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    if(abs(localRay.d.z) < 1e-10) { return false; } 

    float t = localRay.o.z / -localRay.d.z;
    if (t <= 0.0 || t >= ray.tNear) { return false; }
    
    float u = (localRay.o.x + localRay.d.x * t) + 0.5;
    float v = (localRay.o.y + localRay.d.y * t) + 0.5;
    
    if(bounded && (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)) { return false; }   
    
    ray.tNear = t;
    SetRayFlag(ray, kFlagsBackfacing, localRay.o.z < 0.0);
    hit.n = transpose(transform.rot) * vec3(0.0, 0.0, 1.0) * transform.sca;
    hit.kickoff = 1e-4;
    hit.uv = vec2(u, v);
    
    return true;
}