#version 330			// Shader 3.3
precision highp float;	// normal floats, makes no difference on desktop computers


const vec3 La = vec3(0.2f, 0.4f, 0.6f);
const vec3 Le = vec3(2.5f, 2.5f, 1.8f);
const vec3 lightPosition = vec3(0.2f, 0.2f, 0.2f);
const vec3 ka = vec3(0.5f, 0.5f, 0.5f);
const float shininess = 100.0f;
const int maxdepth = 5;
const float epsilon = 0.01f;

struct Hit {
    float t;
    vec3 pos, normal;
    int mat;
};

struct Ray {
    vec3 start, dir, weight;
};

const int objFaces = 12;
uniform vec3 wEye, v[20];
uniform int planes[objFaces * 3];
uniform vec3 kd[2], ks[2], F0;
uniform float a, b, c;

void getObjPlane(int i, float scale, out vec3 p, out vec3 normal, vec3 trans){
    vec3 p1 = v[planes[3 * i] - 1] + trans, p2 = v[planes[3 * i + 1] - 1] + trans, p3 = v[planes[3 * i + 2] - 1] + trans;
    normal = cross(p2 - p1, p3 - p1);
    if (dot(p1, normal) < 0) normal = -normal;
    p = p1 * scale;
}

struct Sphare{
    vec3 center;
    float radius;
};

Hit intersactSphere(Ray ray, Hit besthit){
    Sphare object;
    object.radius = 0.3f;
    object.center = vec3(0,0,0);
    Hit hit;
    hit.t = -1;
    vec3 dist = ray.start - object.center;
    float a = dot(ray.dir, ray.dir);
    float b = dot(dist, ray.dir) * 2.0f;
    float c = dot(dist, dist) - object.radius * object.radius;
    float discr = b * b - 4.0f * a * c;
    if (discr < 0) return besthit;
    float sqrt_discr = sqrt(discr);
    float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
    float t2 = (-b - sqrt_discr) / 2.0f / a;
    if (t1 <= 0) return besthit;
    hit.t = (t2 > 0) ? t2 : t1;
    hit.pos = ray.start + ray.dir * hit.t;
    hit.normal = (hit.pos - object.center) / object.radius;
    hit.mat = 2;
    if (hit.t < besthit.t || besthit.t < 0){
        return hit;
    }
    return besthit;
}

Hit intersectGoldenThing(Ray ray, Hit besthit, int mat) {
    Hit tmpSphere;
    tmpSphere.t = -1;
    tmpSphere = intersactSphere(ray, tmpSphere);

    if (tmpSphere.t > 0) {
        ray.start = hit.pos + hit.normal * epsilon;
    }
    Hit hit;
    hit.t = -1;
    float A = a * ray.dir.x * ray.dir.x + b * ray.dir.y * ray.dir.y;
    float B = 2 * a * ray.start.x * ray.dir.x + 2 * b * ray.start.y * ray.dir.y - c * ray.dir.z;
    float C = a * ray.start.x * ray.start.x + b * ray.start.y * ray.start.y - c * ray.start.z;
    float disc = B * B - 4 * A * C;
    if (disc < 0) {
        return besthit;
    }
    float sqrt_discr = sqrt(discr);
    float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
    float t2 = (-b - sqrt_discr) / 2.0f / a;

    vec3 pt1 = ray.start + ray.dir * t1;
    vec3 pt2 = ray.start + ray.dir * t2;

    if (length(pt1) > 0.3 && length(pt2) > 0.3) {
        return besthit;
    } else if (length(pt1) > 0.3) {
        hit.t = t2;
        hit.pos = p2;
    } else if (length(pt2) > 0.3) {
        hit.t = t1;
        hit.pos = p1;
    } else {
        if (t1 < 0 && t2 < 0) {
            return besthit;
        } else if (t1 < 0) {
            hit.t = t2;
            hit.pos = pt2;
        } else if (t2 < 0) {
            hit.t = t1;
            hit.pos = pt1;
        }
    }

    vec3 dx = vec3(1, 0, (2 * a * hit.pos.x) / c);
    vec3 dy = vec3(0, 1, (2 * b * hit.pos.y) / c);
    hit.normal = normalize(cross(dx, dy));

    hit.mat = mat;

    Hit tmpSphere;
    tmpSphere.t = -1;
    tmpSphere = intersactSphere(ray, tmpSphere);

    if ((hit.t < besthit.t || besthit.t < 0) && tmpSphere.t > 0){
        return hit;
    }
    return besthit;
}

Hit intersectConvexPolyhedronFilled(Ray ray, Hit hit, float scale, int mat) {
    for (int i = 0; i < objFaces; ++i) {
        vec3 p1, normal;
        //select active plane
        getObjPlane(i, scale, p1, normal, vec3(0, 1.5f, 0));

        //calc hit t.
        float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
        //not hiited
        if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;

        vec3 pintersect = (ray.start + ray.dir * ti);

        bool outside = false;
        for (int j = 0; j < objFaces; ++j) {
            if (i == j) continue;

            vec3 p11, n;
            getObjPlane(j, scale, p11, n, vec3(0, 1.5f, 0));
            if (dot(n, pintersect - p11) > 0) {
                outside = true;
                break;
            }
        }

        if (!outside) {
            Hit tmpSphere;
            tmpSphere.t = -1;
            tmpSphere = intersactSphere(ray, tmpSphere);
            if (tmpSphere.t > 0) {
                hit.t = ti;
                hit.pos = pintersect;
                hit.normal = normalize(normal);
                hit.mat = mat;
            }
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
            bool portal = true;
            for(int j = 0; j < objFaces; j++){
                if (i == j) continue;
                vec3 p11, n;
                getObjPlane(j, scale, p11, n);
                hit.t = ti;
                hit.pos = pintersect;
                hit.normal = normalize(normal);
                if (abs(dot(n, pintersect - p11)) < 0.1f) {
                    hit.mat = mat;
                    portal = false;
                    break;
                }
            }
            if (portal) {
                hit.mat = 3;
            }
        }
    }
    return hit;
}

Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;
    bestHit = intersectConvexPolyhedronNotFilled(ray, bestHit, 1, 1);
    bestHit = intersectGoldenThing(ray, bestHit, 2);
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
            float phi = 72.0f * 3.14 / 180;
            ray.start = hit.pos + hit.normal * epsilon;
            ray.dir = reflect(ray.dir, hit.normal);
            vec3 tmpPoint = ray.start + ray.dir;

            ray.start = ray.start * cos(phi) + cross(hit.normal, ray.start) * sin(phi) + hit.normal * hit.normal * ray.start * (1 - cos(phi));
            tmpPoint = tmpPoint * cos(phi) + cross(hit.normal, tmpPoint) * sin(phi) + hit.normal * hit.normal * tmpPoint * (1 - cos(phi));
            ray.dir = normalize(tmpPoint - ray.start);
            continue;
        }
        ray.weight *= F0 + (vec3(1, 1, 1) - F0) * pow(dot(-ray.dir, hit.normal), 5);
        ray.start = hit.pos + hit.normal * epsilon;
        ray.dir = reflect(ray.dir, hit.normal);
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