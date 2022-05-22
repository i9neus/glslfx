// #include "BxDF.glsl"

float SampleLambertianBDRF(inout Ray ray, in HitCtx hit, in int sampleIdx, int depth, inout vec3 L)
{        
    vec2 xi = Rand().xy;
    
    // Sample the Lambertian direction
    vec3 r = vec3(SampleUnitDisc(xi.xy), 0.0);
    r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));
    
    // Transform it to world space
    mat3 basis = CreateBasis(hit.n);     
        
    // Create the ray from the sampled BRDF direction
    CreateRay(ray, PointAt(ray),
                           basis * r, 
                           //(IsBackfacing(ray) ? -hit.n : hit.n) * hit.kickoff,
                           // FIXME: Why does this break dielectrics? 
                           hit.n * hit.kickoff,
                           kTwoPi * ray.weight,
                           r.z / kPi,
                           InheritFlags(ray) | kFlagsScattered);    

    return r.z / kPi;
}

float EvaluateLambertianBRDF(in vec3 d, in vec3 n, out float weight)
{
    weight = dot(d, n) / kPi;
    return weight;
}