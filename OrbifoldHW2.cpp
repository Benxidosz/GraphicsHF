//=============================================================================================
// Mintaprogram: Zöld háromszág. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szabó Bence Sándor
// Neptun : NQB6UO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include <fstream>
#include <sstream>
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330
    precision highp float;

    uniform vec3 wLookAt, wRight, wUp;          // pos of eye

    layout(location = 0) in vec2 cCamWindowVertex;    // Attrib Array 0
    out vec3 p;

    void main() {
        gl_Position = vec4(cCamWindowVertex, 0, 1);
        p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
    }
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
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

    void getObjPlane(int i, float scale, out vec3 p, out vec3 normal){
        vec3 p1 = v[planes[3 * i] - 1], p2 = v[planes[3 * i + 1] - 1], p3 = v[planes[3 * i + 2] - 1];
        normal = cross(p2 - p1, p3 - p1);
        if (dot(p1, normal) < 0) normal = -normal;
        p = p1 * scale;
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
                float phi = 72.0f * 3.14 / 180;
                ray.start = hit.pos + hit.normal * epsilon;
                /*ray.start.x = ray.start.x * cos(phi) - ray.start.y * sin(phi);
                ray.start.y = ray.start.x * sin(phi) + ray.start.y * cos(phi);*/

                ray.dir = reflect(ray.dir, hit.normal);
                ray.dir = ray.dir * cos(phi) + cross(hit.normal, ray.dir) * sin(phi) + hit.normal * hit.normal * ray.dir * (1 - cos(phi));
                ray.dir = normalize(ray.dir);
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
)";

struct Camera {
    vec3 eye, lookat, right, pvup, rvup;
    float fov = 45 * (float)M_PI / 180;

    Camera() : eye(0,1,1), pvup(0,0,1), lookat(0,0,0) { set(); }
    void set() {
        vec3 w = eye - lookat;
        float f = length(w);
        right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
        rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
    }
    void Animate(float t) {
        float r = sqrt(eye.x * eye.x + eye.y * eye.y);
        eye = vec3(r * cosf(t) + lookat.x, r * sinf(t) + lookat.y, eye.z);
        set();
    }
};

GPUProgram gpuProgram(false);
Camera camera;
bool animate = true;

float F(float n, float k) {
    return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
}

// Initialization, create an OpenGL context
void onInitialization() {
    unsigned int vao, vbo;
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	float vertices[] = {
	        -1, -1,
            1, -1,
            1, 1,
            -1, 1
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    gpuProgram.create(vertexSource, fragmentSource, "outColor");

	const float g = 0.618f, G = 1.618f;
    std::vector<vec3> v = {
	    vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G), vec3(G, 0, g), vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g),
	    vec3(g,G,0), vec3(-g,G,0), vec3(-g,-G,0), vec3(g,-G,0), vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1,-1,1),
	    vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1)
	 };

	for (int i = 0; i < v.size(); ++i) gpuProgram.setUniform(v[i], "v[" + std::to_string(i) + "]");

    std::vector<int> planes = {
        1,2,16, 1,13,9, 1,14,6, 2,15,11, 3,4,18, 3,17,12, 3,20,7, 19,10,9, 16,12,17, 5,8,18, 14,10,19, 6,7,20
    };
    for (int i = 0; i < planes.size(); ++i) gpuProgram.setUniform(planes[i], "planes[" + std::to_string(i) + "]");

    gpuProgram.Use();
    gpuProgram.setUniform(vec3(0.1f,0.2f,0.3f), "kd[0]");
    gpuProgram.setUniform(vec3(0.5f, 0.5f, 0.5f), "kd[1]");
    gpuProgram.setUniform(vec3(5,5,5), "ks[0]");
    gpuProgram.setUniform(vec3(1,1,1), "ks[1]");
    gpuProgram.setUniform(vec3(F(0.17,3.1), F(0.35,2.7), F(1.5,1.9)), "F0");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

    gpuProgram.setUniform(camera.eye, "wEye");
    gpuProgram.setUniform(camera.lookat, "wLookAt");
    gpuProgram.setUniform(camera.right, "wRight");
    gpuProgram.setUniform(camera.rvup, "wUp");

	glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 4 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 's') animate = false;
    if (key == 'S') animate = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (animate) camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
	glutPostRedisplay();
}
