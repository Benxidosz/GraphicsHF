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
    Camera2D() : wCenter(0, 0), wSize(300, 300) { }

    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); };
    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); };

    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }

    vec2 getSize() {
        return wSize;
    }
};

GPUProgram gpuProgram;
Camera2D camera;

class Graph {
    unsigned int vaoNodes, vboNodes;
    unsigned int vaoEdges, vboEdges;
    std::vector<vec3> nodes;
    std::vector<vec4> edges;

    void generateNodes() {
        vec2 wSize = camera.getSize();
        while (nodes.size() < 50) {
            float randX = rand() % ((int) wSize.x - 20) - (int) wSize.x / 2 + 10;
            float randY = rand() % ((int) wSize.y - 20) - (int) wSize.y / 2 + 10;

            bool near = false;

            for (int i = 0; i < nodes.size(); ++i) {
                vec2 tmp = vec2(nodes[i].x, nodes[i].y);
                if (sqrt(pow(tmp.x - randX, 2) + pow(tmp.y - randY, 2)) < 20) {
                    near = true;
                    printf("x1 %f x2 %f y1 %f y2 %f d: %f\n", tmp.x, randX, tmp.y, randY, dot(tmp,vec2(randX, randY)));
                    continue;
                }
            }

            if (!near)
                nodes.emplace_back(vec3(randX, randY, sqrt(pow(randX, 2) + pow(randY, 2) + 1)));
        }
    }

    void generateEdges() {
        int maxEdgenum = (50 * 49) / 2;
        while ((float)edges.size() / (float)maxEdgenum < 0.05) {
            int n1 = rand() % 50;
            int n2 = rand() % 50;

            vec2 n1Pos = vec2(nodes[n1].x, nodes[n1].y);
            vec2 n2Pos = vec2(nodes[n2].x, nodes[n2].y);

            if (n1 != n2) {
                bool has = false;
                for (auto iter = edges.begin(); iter != edges.end(); iter++) {
                    vec4 tmp = *iter;

                    if ((tmp.x == n1Pos.x and tmp.y == n1Pos.x and tmp.z == n2Pos.x and tmp.w == n2Pos.x)
                        or (tmp.x == n2Pos.x and tmp.y == n2Pos.x and tmp.z == n1Pos.x and tmp.w == n1Pos.x)) {
                        has = true;
                        continue;
                    }
                }
                if (!has)
                    edges.emplace_back(n1Pos.x, n1Pos.y , n2Pos.x, n2Pos.y);
            }
        }
    }
public:
    void Create() {
        //Generate Nodes location.
        generateNodes();

        //Generate edges.
        generateEdges();

        //uploadable (x, y)
        vec2 upload[50];

        for (int i = 0; i < 50; ++i)
            upload[i] = vec2(nodes[i].x, nodes[i].y);

        //Make Nodes vao/vbo
        glGenVertexArrays(1, &vaoNodes);
        glBindVertexArray(vaoNodes);

        glGenBuffers(1, &vboNodes);
        glBindBuffer(GL_ARRAY_BUFFER, vboNodes);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * 50, upload, GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        //Make Edge vao/vbo
        glGenVertexArrays(1, &vaoEdges);
        glBindVertexArray(vaoEdges);

        glGenBuffers(1, &vboEdges);
        glBindBuffer(GL_ARRAY_BUFFER, vboEdges);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec4) * edges.size(), &edges[0], GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        mat4 VPTransform = camera.V() * camera.P();
        gpuProgram.setUniform(VPTransform, "MVP");

        //draw Edges
        gpuProgram.setUniform(vec3(1,1,0), "color");
        glLineWidth(2.0f);
        glBindVertexArray(vaoEdges);
        glDrawArrays(GL_LINES, 0 , edges.size());

        //draw Nodes
        gpuProgram.setUniform(vec3(1,0,0), "color");
        glPointSize(10.0f);
        glBindVertexArray(vaoNodes);
        glDrawArrays(GL_POINTS, 0 , 50);
    }
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    srand(time(0));

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");

	graph.Create();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	graph.Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') camera.Pan(vec2(1, 0));
	if (key == 'w') camera.Pan(vec2(0, 1));
	if (key == 'a') camera.Pan(vec2(-1, 0));
	if (key == 's') camera.Pan(vec2(0, -1));
	glutPostRedisplay();
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
