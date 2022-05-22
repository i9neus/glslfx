// Decompose a 2x2 matrix into its constituent eigenvalues and eigenvectors
bool Eigendecompose(in mat2 A, out vec2 v0, out vec2 v1, out float l0, out float l1)
{
    if(abs(A[0][0]) < 1e-10 && abs(A[0][1]) < 1e-10 && abs(A[1][0]) < 1e-10 && abs(A[1][1]) < 1e-10) { return false; }
    
    // Eigenvalues
    float tr = 0.5 * (A[0][0] + A[1][1]);    
    float det = determinant(A);    
    l0 = tr + sqrt(tr*tr - det);
    l1 = tr - sqrt(tr*tr - det);
    
    // For a diagonal matrix, the eigenvectors are the two rows of the input matrix
    if(abs(A[1][0]) < 1e-10 && abs(A[0][1]) < 1e-10)
    {
        v0 = normalize(A[0]);
        v1 = normalize(A[1]);
    }
    else
    {    
       v0 = normalize((A[0][1] > A[1][0]) ? vec2(1, (l0 - A[0][0]) / A[0][1]) : vec2((l0 - A[1][1]) / A[1][0], 1));
       v1 = normalize((A[0][1] > A[1][0]) ? vec2(1, (l1 - A[0][0]) / A[0][1]) : vec2((l1 - A[1][1]) / A[1][0], 1));
    }
    
    return true;
    
}