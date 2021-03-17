//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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
// Nev    : Szab� Bence S�ndor
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

class Camera2D {
    vec2 wCenter;
    vec2 wSize;

public:
    Camera2D() : wCenter(0, 0), wSize(200, 200) { }

    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); };
    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); };

    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Camera2D camera;
const int nTesselatedVertices = 100;

class Curve{
    unsigned int vboCurve, vaoCurve;
    unsigned int vboCtrlPoints, vaoCtrlPoints;
protected:
    std::vector<vec2> wCtrlPoints;
public:
    void create() {
        glGenVertexArrays(1, &vaoCurve);	// get 1 vao id
        glBindVertexArray(vaoCurve);		// make it active
        glGenBuffers(1, &vboCurve);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);

        glGenVertexArrays(1, &vaoCtrlPoints);	// get 1 vao id
        glBindVertexArray(vaoCtrlPoints);		// make it active
        glGenBuffers(1, &vboCtrlPoints);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
        glEnableVertexAttribArray(0);  // AttribArray 1
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
    }

    ~Curve() {
        glDeleteBuffers(1, &vboCurve);
        glDeleteBuffers(1, &vboCtrlPoints);
        glDeleteVertexArrays(1, &vaoCurve);
        glDeleteVertexArrays(1, &vaoCtrlPoints);
    }

    virtual vec2 r(float t) = 0;
    virtual float tStart() = 0;
    virtual float tEnd() = 0;

    virtual void AddControlPoint(float cX, float cY) {
        vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        wCtrlPoints.push_back(vec2(wVertex.x, wVertex.y));
    }

    int PickControlPoint(float cX, float cY) {
        vec4 hVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        vec2 wVertex = (hVertex.x, hVertex.y);

        for (int i = 0; i < wCtrlPoints.size(); ++i) {
            if (dot(wCtrlPoints[i] - wVertex, wCtrlPoints[i] - wVertex) < 0.1)
                return i;
        }
        return -1;
    }

    void MoveControlPoint(int p, float cX, float cY) {
        vec4 hVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        wCtrlPoints[p] = vec2(hVertex.x, hVertex.y);
    }

    void Draw() {
        mat4 VPTransform = camera.V() * camera.P();
        gpuProgram.setUniform(VPTransform, "MVP");

        if (wCtrlPoints.size() > 0) {
            glBindVertexArray(vaoCtrlPoints);
            glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
            glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * sizeof(vec2), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
            gpuProgram.setUniform(vec3(1, 0, 0), "color");
            glPointSize(10.0f);
            glDrawArrays(GL_POINTS, 0 , wCtrlPoints.size());
        }

        if (wCtrlPoints.size() >= 2) {
            std::vector<vec2> vertexData;
            for (int i = 0; i < nTesselatedVertices; ++i) {
                float tNormalized = (float)i / (nTesselatedVertices - 1);
                float t = tStart() + (tEnd() - tStart()) * tNormalized;
                vertexData.push_back(r(t));
            }

            glBindVertexArray(vaoCurve);
            glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
            glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec2), &vertexData[0], GL_DYNAMIC_DRAW);
            gpuProgram.setUniform(vec3(1, 1, 0), "color");
            glLineWidth(2.0f);
            glDrawArrays(GL_LINE_STRIP, 0 , nTesselatedVertices);
        }
    }
};

class BezierCurve : public Curve {
    float B(int i, float t) {
        int n = wCtrlPoints.size() - 1;
        float choose = 1;
        for (int j = 0; j < i; ++j)
            choose *= (float)(n - j + 1) / j;
        return  choose * pow(t, i) * pow(1 - t, n - i);
    }
public:
    float tStart() override { return 0;}
    float tEnd() override { return 1;}

    vec2 r(float t) override {
        vec2 wPoint(0, 0);
        for (int i = 0; i <wCtrlPoints.size(); ++i)
            wPoint = wPoint + wCtrlPoints[i] * B(i, t);
        return wPoint;
    }
};

Curve * curve;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2.0f);

    curve = new BezierCurve();

    curve->create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	curve->Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
