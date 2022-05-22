// A spatial transform decomposed as a translation vector, rotation matrix and a scalar scale factor.
// TODO: Add support for non-linear scaling
struct Transform
{
    vec3 trans;
    mat3 rot;
    float sca;
};

mat3 ScaleMat3(float scale)
{
    float invScale = 1.0f / scale;
	return mat3(vec3(invScale, 0.0, 0.0),
			vec3(0.0, invScale, 0.0),
			vec3(0.0, 0.0, invScale));
}

mat3 RotXMat3(float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(1.0, 0.0, 0.0),
			vec3(0.0, cosTheta, -sinTheta),
			vec3(0.0, sinTheta, cosTheta));
}

mat3 RotYMat3(const float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(cosTheta, 0.0, sinTheta),
			vec3(0.0, 1.0, 0.0),
			vec3(-sinTheta, 0.0, cosTheta));
}

mat3 RotZMat3(const float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(cosTheta, -sinTheta, 0.0),
			vec3(sinTheta, cosTheta, 0.0),
			vec3(0.0, 0.0, 1.0));
}

Transform CompoundTransform(vec3 trans, vec3 rot, float scale)
{
    Transform t;
    t.rot = Identity();
    t.sca = scale;
    t.trans = trans;

    if (rot.x != 0.0) { t.rot *= RotXMat3(rot.x); }
    if (rot.y != 0.0) { t.rot *= RotYMat3(rot.y); }
    if (rot.z != 0.0) { t.rot *= RotZMat3(rot.z); }

    if (scale != 1.0f) { t.rot *= ScaleMat3(scale); }
    
    return t;
}

Transform IdentityTransform()
{
    Transform t;
    t.rot = Identity();
    t.sca = 1.0;
    t.trans = kZero;
    return t;
}

// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
mat3 CreateBasis(vec3 n)
{
    float s = Sign(n.z);
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    
    return mat3(vec3(1.0f + s * n.x * n.x * a, s * b, -s * n.x),
                vec3(b, s + n.y * n.y * a, -n.y),
                n);
}

mat3 CreateBasis(vec3 n, vec3 up)
{
    vec3 tangent = normalize(cross(n, up));
    vec3 cotangent = cross(tangent, n);

    return transpose(mat3(tangent, cotangent, n));
}