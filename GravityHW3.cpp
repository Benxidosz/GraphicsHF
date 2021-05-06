//=============================================================================================
// Mintaprogram: Zöld höromszög. Ervenyes 2019. osztol.
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
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

struct Camera {
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic

    virtual mat4 V() = 0;
    virtual mat4 P() = 0;
};

struct PerspectiveCamera : Camera {
    PerspectiveCamera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }
};

struct OrthogonalCamera : Camera {
    OrthogonalCamera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 1000;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1,                      0,                0,                               0,
                    0,                      1,                0,                               0,
                    0,                      0,                -2.0f / (bp - fp),               0,
                    0,                      0,                -(bp - fp + 2 * fp) / (bp - fp), 1);
    }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

vec4 getRandomColor() {
    return vec4((float)(rand() % 255) / 255,(float)(rand() % 255) / 255,(float)(rand() % 255) / 255, 1.0f);
}

class BasicSphereTexture : public Texture {
public:
    BasicSphereTexture(vec4 color) : Texture() {
        std::vector<vec4> image(10 * 10);
        for (int x = 0; x < 10; x++)
            for (int y = 0; y < 10; y++) {
                image[y * 10 + x] = color;
            }
        create(10, 10, image, GL_NEAREST);
    }
};

struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    Texture *          texture;
    vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

class SheetShader : public PhongShader {
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
};

class Geometry {

protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    struct VertexData {
        vec3 position, normal;
        vec2 texcoord;
    };

    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        vtxData.texcoord = vec2(u, v);
        Dnum2 X, Y, Z;
        Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        eval(U, V, X, Y, Z);
        vtxData.position = vec3(X.f, Y.f, Z.f);
        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = cross(drdU, drdV);
        return vtxData;
    }

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;	// vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

class Sphere : public ParamSurface {
public:
    Sphere() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
        X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
    }
};

float globalMass = 0.001f;

struct Mass {
    vec3 pos;
    float m;
    Mass(vec2 pos) : pos(pos.x, pos.y, 1), m(globalMass) {
        globalMass += 0.001f;
    }
};

std::vector<Mass> masses;
vec3 gravity(0, 0, -20);
float epsz = 0.04f;

float r0 = 2 * 0.005;

class SheetGeo : public ParamSurface {
public:
    SheetGeo() { create(); }

    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f;
        V = V * 2.0f;
        X = U - 1.0f;
        Y = V - 1.0f;
        Z = 0.0f;
        for (Mass& mass : masses) {
            Dnum2 r = (Dnum2(mass.pos.x) - X) * (Dnum2(mass.pos.x) - X) + (Dnum2(mass.pos.y) - Y) * (Dnum2(mass.pos.y) - Y);
            Z = Z - (Dnum2(mass.m) / (r + r0));
        }
    }
};

struct Object {
    Shader *   shader;
    Material * material;
    Texture *  texture;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;

    Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
            scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }

    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    virtual bool isAlive() {
        return true;
    }

    virtual void Animate(float tstart, float tend) { }
};

struct Ball : public Object{
    vec3 velocity;
    vec3 logicPos;
    bool active;
public:
    Ball() : Object(new PhongShader(), new Material, new BasicSphereTexture(getRandomColor()), new Sphere()), velocity(0, 0, 0), active(false) {
        scale = vec3(0.03f, 0.03f, 0.03f);
        translation = logicPos = vec3(-0.97f, -0.97f, 0.97);

        material->kd = vec3(0.6f, 0.4f, 0.2f);
        material->ks = vec3(4, 4, 4);
        material->ka = vec3(0.1f, 0.1f, 0.1f);
        material->shininess = 100;
    }

    void setVelocity(vec2 v) {
        velocity.x = v.x;
        velocity.y = v.y;
        velocity.z = 0;
    }

    void Animate(float tstart, float tend) override {
        if (active) {
            float dt = tend - tstart;
            Dnum2 X(logicPos.x, vec2(1, 0));
            Dnum2 Y(logicPos.y, vec2(0, 1));

            Dnum2 Z = 0.0f;
            for (Mass &mass : masses) {
                Dnum2 r = (Dnum2(mass.pos.x) - X) * (Dnum2(mass.pos.x) - X) +
                          (Dnum2(mass.pos.y) - Y) * (Dnum2(mass.pos.y) - Y);
                Z = Z - (Dnum2(mass.m) / (r + r0));
            }

            vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
            vec3 normal = normalize(cross(drdU, drdV));

            //force calc
            vec3 force = gravity - dot(gravity, normal) * normal;

            velocity = velocity + (force * dt);

            logicPos = logicPos + vec3(velocity.x, velocity.y, 0) * dt;
            if (logicPos.x < -1)
                logicPos.x = 1;
            else if (logicPos.x > 1)
                logicPos.x = -1;

            if (logicPos.y < -1)
                logicPos.y = 1;
            else if (logicPos.y > 1)
                logicPos.y = -1;

            Z = 0.0f;
            for (Mass &mass : masses) {
                Dnum2 r = (Dnum2(mass.pos.x) - X) * (Dnum2(mass.pos.x) - X) +
                          (Dnum2(mass.pos.y) - Y) * (Dnum2(mass.pos.y) - Y);
                Z = Z - (Dnum2(mass.m) / (r + r0));
            }

            drdU.z = Z.d.x;
            drdV.z = Z.d.y;
            normal = normalize(cross(drdU, drdV));

            translation = logicPos;

            translation.x = X.f;
            translation.y = Y.f;
            translation.z = Z.f;

            translation = translation + normal * 0.03;
        }
    }

    virtual bool isAlive() {
        for (Mass & mass : masses) {
            if (length(logicPos - mass.pos) < epsz)
                return false;
        }
        return true;
    }
};

vec4 qmul (vec4 q1, vec4 q2) {
    vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
    vec3 sol = d2 * q1.w + d1 * q2.w + cross(d1, d2);
    return {sol.x, sol.y, sol.z, q1.w * q2.w - dot (d1, d2)};
}

vec3 Rotate(vec3 u, vec4 q) {
    vec4 qinv(-q.x, -q.y, -q.z, q.w);
    vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
    return {qr.x, qr.y, qr.z};
}

Ball * activeBall;
Object* sheet;

class Scene {
    std::vector<Object *> objects;
    Camera* camera;
    OrthogonalCamera* oCamera;
    PerspectiveCamera* sCamera;

    std::vector<Light> lights;
    std::vector<vec3> startLight;
    bool perspective = false;
public:
    void Build() {
        Material* material = new Material();
        material->kd = vec3(0.6f, 0.4f, 0.2f);
        material->ks = vec3(0.5, 0.5, 0.5);
        material->ka = vec3(0.1f, 0.1f, 0.1f);
        material->shininess = 100;
        sheet = new Object(new SheetShader(), material,
                                   new BasicSphereTexture(vec4(120.0f / 255.0f, 112.0f / 255.0f, 66.0f / 255.0f)), new SheetGeo);

        objects.push_back(sheet);

        activeBall = new Ball();
        objects.push_back(activeBall);
        // Orthogonal Camera
        oCamera = new OrthogonalCamera();
        oCamera->wEye = vec3(0, 0, 10);
        oCamera->wLookat = vec3(0, 0, 0);
        oCamera->wVup = vec3(0, 1, 0);
        camera = oCamera;

        //Perspective camera
        sCamera = new PerspectiveCamera();
        sCamera->wEye = vec3(-0.97f, -0.97f, 0.97);
        sCamera->wLookat = vec3(0, 0, 0);
        sCamera->wVup = vec3(0, 0, 1);

        // Lights
        lights.resize(2);
        lights[0].wLightPos = vec4(0, 2, 5, 1);
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(1.0f, 1.0f, 1.0f);
        startLight.emplace_back(0, 2, 5);

        lights[1].wLightPos = vec4(0, -2, 5, 1);
        lights[1].La = vec3(0.1f, 0.1f, 1);
        lights[1].Le = vec3(1.0f, 1.0f, 1.0f);
        startLight.emplace_back(0, -2, 5);
    }

    void Render() {
        RenderState state;
        state.wEye = camera->wEye;
        state.V = camera->V();
        state.P = camera->P();
        state.lights = lights;
        for (Object * obj : objects) obj->Draw(state);
        for (int i = 0; i < objects.size(); ++i) {
            if (!objects[i]->isAlive()) {
                std::swap(objects[i], objects[objects.size() - 1]);
                objects.pop_back();
            }
        }

    }

    void addBall(Ball* b) {
        objects.push_back(b);
    }

    void Animate(float tstart, float tend) {
        for (Object * obj : objects) obj->Animate(tstart, tend);

        vec4 q(cosf(tstart / 4.0f), sinf(tstart / 4.0f) * (cosf(tstart) / 2.0f), sinf(tstart / 4.0f) * (sinf(tstart) / 2.0f), sinf(tstart / 4.0f) * sqrtf(3.0f / 4.0f));
        vec3 light0 = vec3(lights[0].wLightPos.x, lights[0].wLightPos.y, lights[0].wLightPos.z);
        vec3 light1 = vec3(lights[1].wLightPos.x, lights[1].wLightPos.y, lights[1].wLightPos.z);
        light0 = light0 - startLight[1];
        light1 = light1 - startLight[0];

//        light0 = Rotate(light0, q);
//        light1 = Rotate(light1, q);

        light0 = light0 + startLight[1];
        light1 = light1 + startLight[0];

        lights[0].wLightPos = vec4(light0.x, light0.y, light0.z, 1);
        lights[1].wLightPos = vec4(light1.x, light1.y, light1.z, 1);
    }
    void switchCamera() {
        if (perspective) {
            camera = oCamera;
            perspective = false;
        } else {
            camera = sCamera;
            perspective = true;
        }
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    srand(time(0));
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();									// exchange the two buffers
}

bool space = false;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' ') {
        space = true;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    if (key == ' ') {
        space = false;
    }
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;

    if (state == GLUT_DOWN) {
        if (button == GLUT_LEFT_BUTTON) {
            activeBall->setVelocity(vec2(cX - activeBall->translation.x, cY - activeBall->translation.y));
            activeBall->active = true;
            activeBall = new Ball();
            scene.addBall(activeBall);
            glutPostRedisplay();
        }
        if (button == GLUT_RIGHT_BUTTON) {
            masses.emplace_back(vec2(cX, cY));

            ((ParamSurface*)sheet->geometry)->create();
            glutPostRedisplay();
        }
    }

}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is \94infinitesimal\94
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }

    if (space) {
        scene.switchCamera();
        space = false;
    }
    glutPostRedisplay();
}