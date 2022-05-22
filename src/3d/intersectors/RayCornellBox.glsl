// Intersection test a ray with a 5-sided Cornell box
bool RayCornell(inout Ray ray, out HitCtx hit, in Transform transform)
{
    RayBasic localRay = RayToObjectSpace(ray.od, transform);

    float tNear = ray.tNear;
    vec2 uv;
    vec3 n;
    for(int face = 0; face < 5; face++)
    {    
        int dim = face / 2;
        float side = 2.0 * float(face % 2) - 1.0;
        
        if(abs(localRay.d[dim]) < 1e-10) { continue; }                
               
        float tFace = (0.5 * side - localRay.o[dim]) / localRay.d[dim];
        if (tFace <= 0.0 || tFace >= tNear) { continue; }

        int a = (dim + 1) % 3, b = (dim + 2) % 3;
        vec2 uvFace = vec2((localRay.o[a] + localRay.d[a] * tFace) + 0.5,
                           (localRay.o[b] + localRay.d[b] * tFace) + 0.5);
    
        if(uvFace.x < 0.0 || uvFace.x > 1.0 || uvFace.y < 0.0 || uvFace.y > 1.0) { continue; }
            
        tNear = tFace;
        n = kZero;
        uv = uvFace + vec2(1.0, 0.0) * float(face);
        n[dim] = side;
    }
    
    if(tNear == ray.tNear) { return false; }        
        
    ray.tNear = tNear;
    hit.n = transpose(transform.rot) * n * transform.sca;
    if(dot(n, localRay.d) > 0.0) { hit.n = -hit.n; }
    hit.kickoff = 1e-5;
    SetRayFlag(ray, kFlagsBackfacing, cwiseMax(abs(localRay.o)) < 0.5);  
    
    return true;
}