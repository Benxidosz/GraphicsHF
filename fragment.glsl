#version 330			// Shader 3.3
precision highp float;	// normal floats, makes no difference on desktop computers


const vec3 La = vec3(0.5f, 0.6f, 0.6f);
const vec3 Le = vec3(2.5f, 2.5f, 1.8f);
const vec3 lightPosition = vec3(0.2f, 0.2f, 0.2f);
const vec3 ka = vec3(0.5f, 0.5f, 0.5f);
const float shininess = 100.0f;
const int maxdepth = 5;
const float epsilon = 0.01f;

struct Hit {
    float t;
    vec3 pos, normal, planeNormal;
    int mat;
};

struct Ray {
    vec3 start, dir, weight;
};

const int objFaces = 12;
uniform vec3 wEye, v[20];
uniform int planes[objFaces * 3];
uniform vec3 kd[2], ks[2], F0;

void getObjPlane(int i, float scale, out vec3 p, out vec3 normal){
    vec3 p1 = v[planes[3 * i] - 1], p2 = v[planes[3 * i + 1] - 1], p3 = v[planes[3 * i + 2] - 1];
    normal = cross(p2 - p1, p3 - p1);
    if (dot(p1, normal) < 0) normal = -normal;
    p = p1 * scale + vec3(0, 0, 0.03f);
}

Hit intersectConvexPolyhedronFilled(Ray ray, Hit hit, float scale, int mat) {
    for (int i = 0; i < objFaces; ++i) {
        vec3 p1, normal;
        //select active plane
        getObjPlane(i, scale, p1, normal);

        //calc hit t.
        float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
        //not hiited
        if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;

        vec3 pintersect = ray.start + ray.dir * ti;

        bool outside = false;
        for (int j = 0; j < objFaces; ++j) {
            if (i == j) continue;

            vec3 p11, n;
            getObjPlane(j, scale, p11, n);
            if (dot(n, pintersect - p11) > 0) {
                outside = true;
                break;
            }
        }

        if (!outside) {
            hit.t = ti;
            hit.pos = pintersect;
            hit.normal = normalize(normal);
            hit.mat = mat;
        }
    }
    return hit;
}

Hit intersectConvexPolyhedronNotFilled(Ray ray, Hit hit, float scale, int mat) {
    for (int i = 0; i < objFaces; ++i) {
        vec3 p1, normal;
        //select active plane
        getObjPlane(i, scale, p1, normal);

        //calc hit t.
        float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
        //not hiited
        if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;

        vec3 pintersect = ray.start + ray.dir * ti;

        bool outside = false;
        for (int j = 0; j < objFaces; ++j) {
            if (i == j) continue;

            vec3 p11, n;
            getObjPlane(j, scale, p11, n);
            if (dot(n, pintersect - p11) > 0) {
                outside = true;
                break;
            }
        }

        if (!outside){
            for(int j = 0; j < objFaces; j++){
                if (i == j) continue;
                vec3 p11, n;
                getObjPlane(j, scale, p11, n);
                hit.t = ti;
                hit.pos = pintersect;
                hit.normal = normalize(normal);
                if (abs(dot(n, pintersect - p11)) < 0.1f) {
                    hit.mat = mat;

                } else {
                    hit.mat = 3;
                }
            }
        }
    }
    return hit;
}

Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;
    bestHit = intersectConvexPolyhedronNotFilled(ray, bestHit, sqrt(3), 1);
    bestHit = intersectConvexPolyhedronFilled(ray, bestHit, 0.075f, 2);
    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}

vec3 trace(Ray ray) {
    vec3 outRadiance = vec3(0, 0, 0);
    for (int d = 0; d < maxdepth; ++d) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) break;
        if (hit.mat < 2){
            vec3 lightdir = normalize(lightPosition - hit.pos);
            float cosTheta = dot(hit.normal, lightdir);
            if (cosTheta > 0) {
                vec3 LeIn = Le / dot(lightPosition - hit.pos, lightPosition - hit.pos);
                outRadiance += ray.weight * LeIn * kd[hit.mat] * cosTheta;
                vec3 halfway = normalize(-ray.dir + lightdir);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance += ray.weight * LeIn * ks[hit.mat] * pow(cosDelta, shininess);
            }
            ray.weight *= ka;
            outRadiance += ray.weight;
            return outRadiance;
        }
        if (hit.mat == 3) {
            ray.start.x = ray.start.x * cos(72) - ray.start.y * sin(72);
            ray.start.x = ray.start.x * sin(72) + ray.start.y * cos(72);

            ray.dir = ray.dir * cos(72) + cross(hit.normal, ray.dir) * sin(72) + hit.normal * dot(hit.normal, ray.dir) * (1 - cos(72));
        } else {
            ray.weight *= F0 + (vec3(1, 1, 1) - F0) * pow(dot(-ray.dir, hit.normal), 5);
            ray.start = hit.pos + hit.normal * epsilon;
            ray.dir = reflect(ray.dir, hit.normal);
        }
    }
    outRadiance += ray.weight * La;
    return outRadiance;
}
in vec3 p;

out vec4 outColor;		// computed color of the current pixel

void main() {
    Ray ray;
    ray.start = wEye;
    ray.dir = normalize(p - wEye);
    ray.weight = vec3(1, 1, 1);
    outColor = vec4(trace(ray), 1);
}