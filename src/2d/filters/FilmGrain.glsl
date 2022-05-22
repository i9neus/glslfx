// Film grain is the pattern formed by the heterogeneous distribution of silver particles accross the film medium. 
// https://en.wikipedia.org/wiki/Film_grain
float FilmGrain(in vec2 xyScreen, in float strength, in float frequency)
{  
    //#define kFilmGrainStrength        0.3             // The strength of film grain post-process
    //#define kFilmGrainFreq            2.0             // The relative frequency of the film grain effect. Higher = finer grain.
    
    vec2 uvView = TransformScreenToWorld(xyScreen);
    
    float octaves = 0.0;
    octaves += 1.0 * (texture(iChannel1, vec2(mod(uvView.x * frequency, 1.0), mod(uvView.y * frequency, 1.0))).x);
    octaves += 1.0 * (texture(iChannel1, vec2(mod(uvView.x * 2.0 * frequency, 0.9), mod(uvView.y * 2.0 * frequency, 0.9))).x);
    octaves /= 2.0;
    
    return 1.0 + (octaves - 0.5) * strength;
}