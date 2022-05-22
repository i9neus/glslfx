// Vignetting is an effect caused by composite lenses whereby the image appears darker around the edges.
// Despite being an artefact of lens design, it is often applied deliberately as an artistic effect to 
// frame the image and draw the eye inward toward the center. 
float Vignette(in vec2 fragCoord)
{
    #define kVignetteStrength         0.5            // The strength of the vignette effect
    #define kVignetteScale            1.0            // The scale of the vignette effect
    #define kVignetteExponent         4.0             // The rate of attenuation of the vignette effect
    
    vec2 uv = fragCoord / iResolution.xy;
    uv.x = (uv.x - 0.5) * (iResolution.x / iResolution.y) + 0.5;     
    
    float x = 2.0 * (uv.x - 0.5);
    float y = 2.0 * (uv.y - 0.5);
    
    float dist = sqrt(x*x + y*y) / kRoot2;
    
    return mix(1.0, max(0.0, 1.0 - pow(dist * kVignetteScale, kVignetteExponent)), kVignetteStrength);
}