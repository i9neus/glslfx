// Closed-form approxiation of the error function.
// See 'Uniform Approximations for Transcendental Functions', Winitzki 2003, https://doi.org/10.1007/3-540-44839-X_82
float ErfApprox(float x)
{    
     float a = 8.0 * (kPi - 3.0) / (3.0 * kPi * (4.0 - kPi));
     return sign(x) * sqrt(1.0 - exp(-(x * x) * (4.0 / kPi + a * x * x) / (1.0 + a * x * x)));
}

// The antiderivative of the normalised Gaussian with standard deviation sigma
float AntiderivGauss(float x, float sigma)
{    
    return 0.5 * (1.0 + ErfApprox(x / (sigma * kRoot2)));
}

float CauchyPDF(float x, float D)
{
    return 1.0 / (kPi * D * (1.0 + sqr(x / D)));
}

float CauchyCDF(float x, float D)
{
    return kInvPi * atan(x / D) + 0.5;
}

float FresnelSIntegral(float x)
{
    return 1.2533141373155001 * (sign(x) * 0.5 + (cos(x*x) / (x * 2.5066282746310002) + sin(x*x) / (x*x*x * 5.0132565492620005)));
}

