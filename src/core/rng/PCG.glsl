// *******************************************************************************************************
//    Random number generation
// *******************************************************************************************************

// Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano)
// http://jcgt.org/published/0009/03/02/paper.pdf
uvec4 PCGAdvance()
{
    rngSeed = rngSeed * 1664525u + 1013904223u;
    
    rngSeed.x += rngSeed.y*rngSeed.w; 
    rngSeed.y += rngSeed.z*rngSeed.x; 
    rngSeed.z += rngSeed.x*rngSeed.y; 
    rngSeed.w += rngSeed.y*rngSeed.z;
    
    rngSeed ^= rngSeed >> 16u;
    
    rngSeed.x += rngSeed.y*rngSeed.w; 
    rngSeed.y += rngSeed.z*rngSeed.x; 
    rngSeed.z += rngSeed.x*rngSeed.y; 
    rngSeed.w += rngSeed.y*rngSeed.z;
    
    return rngSeed;
}

// Generates a tuple of canonical random number and uses them to sample an input texture
vec4 Rand(sampler2D sampler)
{
    return texelFetch(sampler, (ivec2(gFragCoord) + ivec2(PCGAdvance() >> 16)) % 1024, 0);
}

// Generates a tuple of canonical random numbers in the range [0, 1]
vec4 Rand()
{
    return vec4(PCGAdvance()) / float(0xffffffffu);
}

// Generates a tuple of canonical random numbers
#define URand PCGAdvance()
ivec4 IRand() { return ivec4(PCGAdvance()); }

// Seed the PCG hash function with the current frame multipled by a prime
void PCGInitialise(uint frame)
{    
    rngSeed = uvec4(20219u, 7243u, 12547u, 28573u) * frame;
}