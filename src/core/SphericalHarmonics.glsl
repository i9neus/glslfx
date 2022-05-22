float GetSHCoeff(in vec3 n, in int band)
{
    switch(band)
    {
         case 0: return 0.2820947917738781;
         case 1: return 0.4886025119029199 * n.x;
         case 2: return 0.4886025119029199 * n.y;
         case 3: return 0.4886025119029199 * -n.z;
         case 4: return 1.0925484305920792 * n.x * n.y;
         case 5: return 1.0925484305920792 * n.y * n.z;
         case 6: return 0.3153915652525200 * (-(n.x*n.x) - n.y*n.y + 2.0 * n.z*n.z);
         case 7: return 1.0925484305920792 * n.z * n.x;
         case 8: return 0.5462742152960396 * (n.x*n.x - n.y*n.y); 
    }
    return 0.0;
}

float GetSHCoeff(in int band)
{
    switch(band)
    {
         case 0: return 0.2820947917738781;
         case 1: return 0.4886025119029199;
         case 2: return 0.4886025119029199;
         case 3: return 0.4886025119029199;
         case 4: return 1.0925484305920792;
         case 5: return 1.0925484305920792;
         case 6: return 0.3153915652525200;
         case 7: return 1.0925484305920792;
         case 8: return 0.5462742152960396; 
    }
    return 0.0;
}