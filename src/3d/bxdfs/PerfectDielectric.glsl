// #include "BxDF.glsl"

float EvaluateSpecularDielectric()
{
    return 0.0;
}

float SampleSpecularDielectric(inout Ray ray, in HitCtx hit)
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
    
    // Calculate the Fresnel coefficient and associated vectors. 
    // Disable reflection for scattered illumiunation
    float F = IsScattered(ray) ? 0.0 : Fresnel(dot(-ray.od.d, n), eta.x, eta.y); 
    vec3 o;  
    
    /*float xi = Rand().x;
    if(xi > F)
    {
        o = refract(ray.od.d, n, eta.x / eta.y);
        n = -n;
    }
    else
    {
        o = reflect(ray.od.d, n);
    } */
    
    ray.weight *= min(1., F);
    o = reflect(ray.od.d, n);

    CreateRay(ray, PointAt(ray), 
                    o, 
                    n * hit.kickoff, 
                    ray.weight, 
                    1e10, 
                    InheritFlags(ray));
                    
    if(IsScattered(ray)) { ray.flags |= kFlagsCausticPath; }
    
    return 1e10;
}