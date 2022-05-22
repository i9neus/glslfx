vec4 SDFSimplexFace(in vec3 p, in vec3 v[4], in vec3 n)
{                             
    for(int i = 0; i < 3; i++)
    {
        vec3 dv = v[(i+1) % 3] - v[i];
        vec3 edgeNorm = normalize(cross(dv, n));
        if(dot(edgeNorm, p - v[i]) > 0.0)
        {            
            float t = clamp((dot(p, dv) - dot(v[i], dv)) / dot(dv, dv), 0.0, 1.0);
            vec3 grad = p - (v[i] + t * dv);
            return vec4(length(grad), grad);
        }
    }
    if(dot(n, p - v[0]) < 0.0) { n = -n; }
    return vec4((dot(p, n) - dot(v[0], n)), n);
}

vec4 SDFDisc(in vec3 p, in vec3 V, in vec3 n, float radius)
{
    // Find the perpendicular point on the plane of the triangle
    vec3 pPlane = p + n * (dot(V, n) - dot(p, n)) - V;
    
    float dist = length(pPlane);
    if(dist > radius) { pPlane *= radius / dist; }    
    pPlane = p - (pPlane + V);
    dist = length(pPlane);
    
    return vec4(dist, pPlane / dist);   
}

vec4 SDFTriangle(in vec3 p, in vec3 V[3], in vec3 n)
{       
    // Localise everything
    V[1] -= V[0]; V[2] -= V[0]; p -= V[0];
    
    // Find the perpendicular point on the plane o  the triangle
    vec3 pPlane = p + n * - dot(p, n);
    
    // Project into the most significant principle component
    vec2 v1, v2, q;
    vec3 ext = max(abs(V[1]), abs(V[2]));
    if(ext.x < ext.y)
    {
        if(ext.x < ext.z) { v1 = V[1].yz; v2 = V[2].yz; q = pPlane.yz; }
        else              { v1 = V[1].xy; v2 = V[2].xy; q = pPlane.xy; }
    }
    else
    {
        if(ext.y < ext.z) { v1 = V[1].xz; v2 = V[2].xz; q = pPlane.xz; }
        else              { v1 = V[1].xy; v2 = V[2].xy; q = pPlane.xy; }
    }   
   
    // Calculate barycentric coordinates
    float beta = -(v1.x * q.y - q.x * v1.y) / (v1.y * v2.x - v1.x * v2.y);
    float alpha = (q.y - v2.y * beta) / v1.y;
    
    // Clamp to the edges of the triangle if out of bounds
    if(alpha < 0.0) 
    {        
         pPlane = V[2] * saturate(dot(V[2], pPlane) / dot(V[2], V[2]));
    }
    else if(beta < 0.0)
    {
        pPlane = V[1] * saturate(dot(V[1], pPlane) / dot(V[1], V[1]));
    }
    else if(alpha + beta > 1.0)
    { 
        V[2] -= V[1];
        pPlane = V[1] + V[2] * saturate(dot(V[2], pPlane - V[1]) / dot(V[2], V[2]));
    }
        
    p -= pPlane;
    float pMag = length(p);
    return vec4(pMag, vec3(p.x, p.y, p.z));
}

bool RaySimplexSDF(inout Ray ray, out HitCtx hit, in Transform transform, in float thickness)
{
    #define kSDFMaxSteps             25
    #define kSDFCutoffThreshold      1e-4
    #define kSDFEscapeThreshold      2.0
    #define kSDFRayIncrement         1.0
    #define kSDFFailThreshold        0.1
    
    #define kNumIntervals            16
    #define kClosedInterval          12
    
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    
    float localMag = length(localRay.d);
    localRay.d /= localMag;
    
    float t = 0.0;
    vec3 p;
    vec4 F;
    bool isBounded = false;
    float surfaceSign = 1.;

    const vec3 V[4] = vec3[](vec3(1,1,1)*.5, vec3(-1,-1,1)*.5, vec3(1,-1,-1)*.5, vec3(-1,1,-1)*.5);
    const int I[12] = int[]( 0, 2, 1, 1, 3, 2, 2, 0, 3, 3, 1, 0);
    const vec3 N[4] = vec3[](-vec3(-0.5773502588, 0.5773502588, -0.5773502588), vec3(-0.5773502588, -0.5773502588, -0.5773502588),
                             -vec3(-0.5773502588, -0.5773502588, 0.5773502588), vec3(-0.5773502588, 0.5773502588, 0.5773502588));                      
 
    int stepIdx;
    float tNear = ray.tNear * localMag;
    for(stepIdx = 0; stepIdx < kSDFMaxSteps; stepIdx++)
    {
        p = localRay.o + localRay.d * t;
        
        F = vec4(kFltMax);
        for(int j = 0; j < 4; j++)
        {              
            int k = j*3;    
            vec4 FFace = SDFTriangle(p, vec3[3](V[I[j*3]], 
                                                  V[I[j*3+1]], 
                                                  V[I[j*3+2]]),
                                                  N[j]);

            FFace.x = FFace.x - thickness;
            if(FFace.x < F.x) 
            { 
                F = FFace;                
            }
        }
      
        // On the first iteration, simply determine whether we're inside the isosurface or not
        if(stepIdx == 0) { surfaceSign = sign(F.x); }
        // Otherwise, check to see if we're at the surface
        else if(abs(F.x) < kSDFCutoffThreshold) { break; }        
        
        if(!isBounded) { if(F.x < kSDFEscapeThreshold) { isBounded = true; } }
        else if(F.x > kSDFEscapeThreshold) { return false; }
        
        t += surfaceSign * F.x * kSDFRayIncrement;
        
        //if(abs(F.x / localMag) < sdfNear.x)
        //            sdfNear = vec2(abs(F.x / localMag), t / localMag);
        
        if(t > tNear) { break; }
        
        p = localRay.o + t * localRay.d;
    }        
    
    if(F.x > kSDFFailThreshold || t > tNear) { return false; }
    
    ray.tNear = t / localMag;
    hit.n = transpose(transform.rot) * normalize(F.yzw) * transform.sca;
    SetRayFlag(ray, kFlagsBackfacing, surfaceSign < 0.0);
    hit.kickoff = 1e-2;

    return true;
}