Ray CreateThinLensCameraRay(vec2 uvScreen, vec3 cameraPos, vec3 cameraLookAt, vec2 resolution, sampler2D sampler)
{     
    // Generate 4 random numbers from a continuous uniform distribution
    vec4 xi = Rand(sampler);
    
    // Define our camera vectors and orthonormal basis
    #define kCameraUp vec3(0.0, 1.0, 0.0)    

    vec3 kCameraForward = cameraLookAt - cameraPos;
    float focalDistance = length(kCameraForward);
    kCameraForward /= focalDistance;
    
    mat4 basis;
    basis[0] = vec4(normalize(cross(kCameraForward, kCameraUp)), 0.0);
    basis[1] = vec4(cross(basis[0].xyz, kCameraForward), 0.0);
    basis[2] = vec4(kCameraForward, 0.0);
    basis[3] = vec4(0.0, 0.0, 0.0, 1.0);
    basis = transpose(basis);
    
    // Define the focal length and F-number depending, either from built-in or user-defined values
    #define kCameraFocalLength 0.050
    #define kCameraFStop 15.0
    #define kCameraSensorSize         0.035 
    #define kCameraAA 1.0
    
    // Solve the thin-lens equation. http://hyperphysics.phy-astr.gsu.edu/hbase/geoopt/lenseq.html
    float d1 = 0.5 * (focalDistance - sqrt(-4.0 * kCameraFocalLength * focalDistance + sqr(focalDistance)));
    float d2 = focalDistance - d1; 
    
    // Generate a position on the sensor, the focal plane, and the lens. This lens will always have circular bokeh
    // but with a few minor additions it's possible to add custom shapes such as irises. We reuse some random numbers
    // but this doesn't really matter because the anti-aliasing kernel is so small that we won't notice the correlation 
    vec2 sensorPos = xi.zx * kCameraSensorSize * kCameraAA / max(resolution.x, resolution.y) + uvScreen * kCameraSensorSize;    
    vec2 focalPlanePos = vec2(sensorPos.x, sensorPos.y) * d2 / d1;    
    vec2 lensPos = SampleUnitDisc(xi.xy) * 0.5 * kCameraFocalLength / kCameraFStop;
    
    // Assemble the ray
    Ray ray;
    ray.od.o = mul3(vec3(lensPos, d1), basis);
    ray.od.d = normalize(mul3(vec3(focalPlanePos, focalDistance), basis) - ray.od.o);
    ray.od.o += cameraPos;
    ray.tNear = kFltMax;
    ray.weight = vec3(1.0, 1.0, 1.0);
    ray.pdf = kFltMax;   

    return ray;    
}