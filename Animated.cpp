//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec3 vc;

    out vec3 color;

	void main() {
        color = vc;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

    in vec3 color;
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

Camera2D camera;
GPUProgram gpuProgram;

class Triangle {
    unsigned int vao;
    float sx, sy;
    vec2 wTranslate;
    float phi;
public:
    Triangle() { Animate(0); }

    void Animate(float t) {
        sx = 10;
        sy = 10;
        wTranslate = vec2(0, 0);
        phi = t;
    }

    void create() {
        glGenVertexArrays(1, &vao);	// get 1 vao id
        glBindVertexArray(vao);		// make it active

        unsigned int vbo[2];		// vertex buffer object
        glGenBuffers(2, vbo);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        // Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
        float vertexCoords[] = { -8, -8,
                             -6, 10,
                             8, -2 };
        glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                     sizeof(vertexCoords),  // # bytes
                     vertexCoords,	      	// address
                     GL_STATIC_DRAW);	// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0,       // vbo -> AttribArray 0
                              2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                              0, NULL); 		     // stride, offset: tightly packedglBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        float vertexColor[] = { 1, 0, 0,
                             0, 1, 0,
                             0, 0, 1 };
        glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                     sizeof(vertexColor),  // # bytes
                     vertexColor,	      	// address
                     GL_STATIC_DRAW);	// we do not change later

        glEnableVertexAttribArray(1);  // AttribArray 0
        glVertexAttribPointer(1,       // vbo -> AttribArray 0
                              3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                              0, NULL); 		     // stride, offset: tightly packed
    }

    mat4 M() {
        mat4 Mscale(    sx,0,0,0,
                    0,    sy,0,0,
                    0,0,0,0,
                    0,0,0,1);
        mat4 Mrotate(cosf(phi), sinf(phi), 0,0,
                    -sinf(phi), cosf(phi), 0,0,
                    0,          0,         0,0,
                    0,          0,         0,1);
        mat4 Mtranslate(1, 0,           0,0,
                        0, 1,           0,0,
                        0, 0,           0,0,
                        wTranslate.x, wTranslate.y,0,1);
        return Mscale * Mrotate * Mtranslate;
    }

    void draw() {
        mat4 MVPtransf = M() * camera.V() * camera.P();
        gpuProgram.setUniform(MVPtransf, "MVP");

        glBindVertexArray(vao);  // Draw call
        glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);
    }
};

class StripLine{
    unsigned int vao, vbo;
    std::vector<float> vertexData;
    vec2 wTranslate;

public:
    void create() {
        glGenVertexArrays(1, &vao);	// get 1 vao id
        glBindVertexArray(vao);		// make it active

        glGenBuffers(1, &vbo);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0,       // vbo -> AttribArray 0
                              2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                              5 * sizeof(float), reinterpret_cast<void*>(0)); 		     // stride, offset: tightly packedglBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glEnableVertexAttribArray(1);  // AttribArray 1
        glVertexAttribPointer(1,       // vbo -> AttribArray 1
                              3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                              5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float))); 		     // stride, offset: tightly packed
    }

    mat4 M() {
        return mat4(1, 0,           0, 0,
                    0, 1,           0, 0,
                    0, 0,           1, 0,
                    wTranslate.x, wTranslate.y,0, 1);
    }

    mat4 Minv() {
        return mat4(1, 0,             0, 0,
                    0, 1,             0, 0,
                    0, 0,             1, 0,
                    -wTranslate.x, -wTranslate.y,0, 1);
    }

    void AddPoint(float cX, float cY) {
        vec4 mVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv() * Minv();

        vertexData.push_back(mVertex.x);
        vertexData.push_back(mVertex.y);
        vertexData.push_back(1);
        vertexData.push_back(1);
        vertexData.push_back(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
    }

    void AddTrans(vec2 wT) {
        wTranslate = wTranslate + wT;
    }

    void draw() {
        if (vertexData.size() > 0) {
            mat4 MVPTransform = M() * camera.V() * camera.P();
            gpuProgram.setUniform(MVPTransform, "MVP");

            glBindVertexArray(vao);  // Draw call
            glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, vertexData.size() / 5 /*# Elements*/);
        }
    }

};

Triangle triangle;
StripLine stripLine;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	triangle.create();
	stripLine.create();

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

    triangle.draw();
    stripLine.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case 'z': camera.Zoom(0.9f); break;
        case 'Z': camera.Zoom(1.1f); break;
        case 'w': camera.Pan(vec2(0,-1)); break;
        case 's': camera.Pan(vec2(0,1)); break;
        case 'a': camera.Pan(vec2(1,0)); break;
        case 'd': camera.Pan(vec2(-1,0)); break;
        case 'k': stripLine.AddTrans(vec2(0,-1)); break;
        case 'i': stripLine.AddTrans(vec2(0,1)); break;
        case 'l': stripLine.AddTrans(vec2(1,0)); break;
        case 'j': stripLine.AddTrans(vec2(-1,0)); break;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        float cX = 2.0f * pX / windowWidth - 1;
        float cY = 1.0f - 2.0f * pY / windowHeight;
        stripLine.AddPoint(cX, cY);
        glutPostRedisplay();
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;
	triangle.Animate(sec);
	glutPostRedisplay();
}
