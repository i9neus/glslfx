float G1(vec3 v, vec3 m, vec3 n, float alpha)
{
    float cosTheta = dot(v, n);
    return step(0.0, dot(v, m) / dot(v, n)) * 
            2.0 / (1.0 + sqrt(1.0 + sqr(alpha) * sqr(sqrt(1.0 - sqr(cosTheta)) / cosTheta)));
}

float EvaluateGGXDielectric(inout Ray ray, in HitCtx hit)
{
    return 0.0;
}

float SampleGGXDielectric(inout Ray ray, in HitCtx hit, float ggxAlpha)
{
    // Figure out what kind of intersection we're doing
    vec2 eta;
    vec3 n;
    if(IsBackfacing(ray)) 
    {
        eta = vec2(kIOR, 1.0);  
        n = -hit.n;
        
        //ray.weight *= EvaluateDielectricAbsorption(ray);
    }
    else
    {
        eta = vec2(1.0, kIOR);
        n = hit.n;
    }      
   
    vec3 xi = Rand().xyz;
    mat3 basis = CreateBasis(n);
    
    // Sample the microsurface normal
    //float thetaM = acos(pow(xi.x, 1.0 / (ggxAlpha + 2.0)));
    float thetaM = atan(ggxAlpha * sqrt(xi.x) / sqrt(1.0 - xi.x));
    float phiM = kTwoPi * xi.y;
    vec3 m = basis * vec3(cos(phiM) * sin(thetaM), sin(phiM) * sin(thetaM), cos(thetaM));
    
    // Calculate the Fresnel coefficient and associated vectors
    float F = Fresnel(dot(-ray.od.d, m), eta.x, eta.y);
    vec3 i = ray.od.d;
    vec3 o;  
    
    if(xi.z > F)
    {        
        o = refract(i, m, eta.x / eta.y);
    }
    else
    {
        o = reflect(i, m);
    }       
    
    float G = G1(i, m, n, ggxAlpha) * G1(o, m, n, ggxAlpha);
    
    // Compute the weight
    float ggxWeight = min(2.0, abs(dot(i, m)) * G / (abs(dot(i, n)) * abs(dot(m, n))));

    CreateRay(ray, PointAt(ray), 
                    o, 
                    ((xi.z > F) ? -m : m) * hit.kickoff, 
                    ray.weight * ggxWeight, 
                    0.0, 
                    InheritFlags(ray));
    
    return 1.0;
}