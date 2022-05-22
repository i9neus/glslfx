float PowerHeuristic(float pdf1, float pdf2)
{
    return pdf1*pdf1 / (pdf1*pdf1 + pdf2*pdf2);
}

float Fresnel(float cosI, float eta1, float eta2)
{
    float sinI = sqrt(1.0 - cosI * cosI);
    float beta = 1.0 - sqr(sinI * eta1 / eta2);
   
    if(beta < 0.0) { return 1.0; }
    
    float alpha = sqrt(beta);
    return (sqr((eta1 * cosI - eta2 * alpha) / (eta1 * cosI + eta2 * alpha)) +
            sqr((eta1 * alpha - eta2 * cosI) / (eta1 * alpha + eta2 * cosI))) * 0.5;
}