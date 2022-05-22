Ray CreateOrthographicCameraRay(vec2 uv, vec2 sensorSize, vec3 cameraPos, vec3 cameraLookAt)
{
    vec3 w = normalize(cameraLookAt - cameraPos);
    mat3 basis = CreateBasis(w, vec3(0.0, 1.0, 0.0));    
    
    Ray ray;
    ray.od.o = cameraPos + vec3(uv * sensorSize, 0.0) * basis;
    ray.od.d = w;
    ray.weight = vec3(1.0);
    return ray;
}
