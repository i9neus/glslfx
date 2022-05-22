// *******************************************************************************************************
//    Colour functions
// *******************************************************************************************************

vec3 Hue(float phi)
{
    float phiColour = 6.0 * phi;
    int i = int(phiColour);
    vec3 c0 = vec3(((i + 4) / 3) & 1, ((i + 2) / 3) & 1, ((i + 0) / 3) & 1);
    vec3 c1 = vec3(((i + 5) / 3) & 1, ((i + 3) / 3) & 1, ((i + 1) / 3) & 1);             
    return mix(c0, c1, phiColour - float(i));
}

// A Gaussian function that we use to sample the XYZ standard observer 
float CIEXYZGauss(float lambda, float alpha, float mu, float sigma1, float sigma2)
{
   return alpha * exp(sqr(lambda - mu) / (-2.0 * sqr(lambda < mu ? sigma1 : sigma2)));
}

vec3 HSVToRGB(vec3 hsv)
{
    return mix(vec3(0.0), mix(vec3(1.0), Hue(hsv.x), hsv.y), hsv.z);
}

vec3 RGBToHSV( vec3 rgb)
{
    // Value
    vec3 hsv;
    hsv.z = cwiseMax(rgb);

    // Saturation
    float chroma = hsv.z - cwiseMin(rgb);
    hsv.y = (hsv.z < 1e-10) ? 0.0 : (chroma / hsv.z);

    // Hue
    if (chroma < 1e-10)        { hsv.x = 0.0; }
    else if (hsv.z == rgb.x)    { hsv.x = (1.0 / 6.0) * (rgb.y - rgb.z) / chroma; }
    else if (hsv.z == rgb.y)    { hsv.x = (1.0 / 6.0) * (2.0 + (rgb.z - rgb.x) / chroma); }
    else                        { hsv.x = (1.0 / 6.0) * (4.0 + (rgb.x - rgb.y) / chroma); }
    hsv.x = fract(hsv.x + 1.0);

    return hsv;
}

vec3 SampleSpectrum(float lambda)
{
	// Here we use a set of fitted Gaussian curves to approximate the CIE XYZ standard observer.
	// See https://en.wikipedia.org/wiki/CIE_1931_color_space for detals on the formula
	// This allows us to map the sampled wavelength to usable RGB values. This code needs cleaning 
	// up because we do an unnecessary normalisation steps as we map from lambda to XYZ to RGB.

	#define kRNorm (7000.0 - 3800.0) / 1143.07
	#define kGNorm (7000.0 - 3800.0) / 1068.7
	#define kBNorm (7000.0 - 3800.0) / 1068.25

	// Sample the Gaussian approximations
	vec3 xyz;
	xyz.x = (CIEXYZGauss(lambda, 1.056, 5998.0, 379.0, 310.0) +
             CIEXYZGauss(lambda, 0.362, 4420.0, 160.0, 267.0) +
             CIEXYZGauss(lambda, 0.065, 5011.0, 204.0, 262.0)) * kRNorm;
	xyz.y = (CIEXYZGauss(lambda, 0.821, 5688.0, 469.0, 405.0) +
             CIEXYZGauss(lambda, 0.286, 5309.0, 163.0, 311.0)) * kGNorm;
	xyz.z = (CIEXYZGauss(lambda, 1.217, 4370.0, 118.0, 360.0) +
             CIEXYZGauss(lambda, 0.681, 4590.0, 260.0, 138.0)) * kBNorm;

	// XYZ to RGB linear transform
	vec3 rgb;
	rgb.r = (2.04159 * xyz.x - 0.5650 * xyz.y - 0.34473 * xyz.z) / (2.0 * 0.565);
	rgb.g = (-0.96924 * xyz.x + 1.87596 * xyz.y + 0.04155 * xyz.z) / (2.0 * 0.472);
	rgb.b = (0.01344 * xyz.x - 0.11863 * xyz.y + 1.01517 * xyz.z) / (2.0 * 0.452);

	return rgb;
}

vec3 Heatmap(float phi)
{
    #define kHeatmapLevels 7
    phi *= float(kHeatmapLevels);
    int i = int(phi);
    if (phi >= float(kHeatmapLevels)) { phi = float(kHeatmapLevels); i = kHeatmapLevels - 1; }
    switch (i)
    {
    case 0: return mix(vec3(0.0),           vec3(0.5, 0.0, 0.5), phi - float(i)); 
    case 1: return mix(vec3(0.5, 0.0, 0.5), vec3(0.0, 0.0, 1.0), phi - float(i));
    case 2: return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), phi - float(i));
    case 3: return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), phi - float(i));
    case 4: return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), phi - float(i));
    case 5: return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), phi - float(i));
    case 6: return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), phi - float(i));
    }

    return kZero;
}

vec3 ColourGradeSkew(in vec3 rgb, in float degree, in float gain)
{
    vec3 hsv = RGBToHSV(saturate(bloomL));
    hsv.x += -sin((hsv.x + degree) * kTwoPi) * gain;
    
    return HSVToRGB(hsv);   
}
