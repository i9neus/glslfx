float SampleQuadLight(inout Ray ray, inout HitCtx hit, inout Transform transform, in vec3 radiance, in int sampleIdx, int depth, inout vec3 L)
{
    // Sample a point on the light 
    vec3 hitPos = PointAt(ray);

    vec2 xi = Rand().xy - 0.5;
    //vec2 xi = vec2(0.0);
    //uint hash = HashOf(uint(gFragCoord.x), uint(gFragCoord.y));
    //vec2 xi = vec2(HaltonBase2(hash + uint(sampleIdx)), HaltonBase3(hash + uint(sampleIdx))) - 0.5;
    
    vec3 lightPos = transform.rot * vec3(xi, 0.) * sqr(transform.sca) + transform.trans;    

    // Compute the normalised extant direction based on the light position local to the shading point
    vec3 extant = lightPos - hitPos;
    float lightDist = length(extant);
    extant /= lightDist;

    // Test if the emitter is behind the shading point
    if (!IsVolumetricBxDF(hit) && dot(extant, hit.n) <= 0.) { return 0.0; }
      
    vec3 lightNormal = normalize(transpose(transform.rot) * vec3(0.0, 0.0, 1.0) * transform.sca);
    float cosPhi = dot(normalize(hitPos - lightPos), lightNormal);
        
    // Test if the emitter is rotated away from the shading point
    if (cosPhi < 0.) { return 0.0; }

    // Compute the projected solid angle of the light        
    float solidAngle = cosPhi * sqr(transform.sca) / max(1e-10, sqr(lightDist));   
        
    // Create the ray from the sampled BRDF direction
    CreateRay(ray, hitPos,
                    extant, 
                    //(IsBackfacing(ray) ? hit.n : hit.n) * hit.kickoff,
                    hit.n * hit.kickoff,
                    ray.weight * radiance * solidAngle,
                    1.0 / max(1e-10, solidAngle),
                    kFlagsLightSample);      

                    
    return ray.pdf;
}

float EvaluateQuadLight(inout Ray ray, inout HitCtx hit, inout Transform transform, in vec3 radiance) 
{        
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    if(abs(localRay.d.z) < 1e-10) { return 0.0; } 
    
    float t = localRay.o.z / -localRay.d.z;
    
    vec2 uv = (localRay.o.xy + localRay.d.xy * t) + 0.5;
    if(cwiseMin(uv) < 0.0 || cwiseMax(uv) > 1.0) { return 0.0; }    
    
    vec3 lightNormal = normalize(transpose(transform.rot) * vec3(0.0, 0.0, 1.0) * transform.sca);
    vec3 lightPos = ray.od.o + ray.od.d * t;
    
    float cosPhi = dot(normalize(ray.od.o - lightPos), lightNormal);
        
    // Test if the emitter is rotated away from the shading point
    if (cosPhi < 0.) { return 0.0; }
    
    float solidAngle = cosPhi * sqr(transform.sca) / max(1e-10, sqr(t));
    
    if(!IsVolumetricBxDF(hit))
    {    
        float cosTheta = dot(hit.n, ray.od.d);
        if (cosTheta < 0.0f)  { return 0.0; }
        
        solidAngle *= cosTheta;   
    }
    
    ray.weight = radiance;
    ray.flags = kFlagsLightSample;
    return 1.0 / max(1e-10, solidAngle);
}

// Stokes' theorem means we can precisely evaluate the irradiance from an arbirary, unoccluded polygon
// "Closed-Form Expressions for Irradiance from Non-Uniform Lambertian Luminaires" (Chen and Arvo)
// https://core.ac.uk/download/pdf/216151544.pdf
vec3 EvaluateQuadLightContour(Ray incident, mat4 emitterMatrix, vec3 emitterRadiance)
{
    mat4 invMatrix = inverse(emitterMatrix);
    vec3 pShade = incident.od.o + incident.od.d * incident.tNear;
    vec3 p[4] = vec3[4](normalize(mul3(vec3(-0.5, -0.5, 0.0), invMatrix) - pShade),
                        normalize(mul3(vec3(0.5, -0.5, 0.0), invMatrix) - pShade),
                        normalize(mul3(vec3(0.5, 0.5, 0.0), invMatrix) - pShade),
                        normalize(mul3(vec3(-0.5, 0.5, 0.0), invMatrix) - pShade));   
    
    float E = 0.0;
    for(int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        E += acos(clamp(dot(p[i], p[j]), -1.0, 1.0)) * normalize(cross(p[i], p[j])).z;
    }    
    
    return E * emitterRadiance * incident.weight / (2.0 * kPi);
}