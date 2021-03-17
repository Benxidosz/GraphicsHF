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

    out vec3 getColor;

	void main() {
        getColor = vc;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
    in vec3 getColor;

	void main() {
        if (getColor != 0)
            outColor = vec4(getColor, 1);
        else
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

bool operator!=(vec3 a, vec3 b) {
    return (a.x != b.x and a.y != b.y and a.z != b.z);
}

bool operator==(vec3 a, vec3 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z);
}

bool operator==(vec4 a, vec4 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z and a.w == b.w);
}

vec4 from2vec2(vec2 a, vec2 b) {
    return vec4(a.x, a.y, b.x, b.y);
}

int nv = 100;

class Node {
    unsigned int vaoCircle[2], vboCircle[2];
    vec3 pos;
    vec3 color[2];

public:
    Node(vec3 pos, vec3 color1, vec3 color2) {
        this->pos = pos;
        color[0] = color1;
        color[1] = color2;
        //Generate circle vertexes.
        vec2 vertices[nv];
        vec2 verticesOut[nv];

        for (int i = 0; i < nv; ++i) {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = vec2(pos.x + 5 * cosf(fi),pos.y + 5 * sinf(fi));
            verticesOut[i] = vec2(pos.x + 2.5 * cosf(fi),pos.y + 2.5 * sinf(fi));
        }

        //Make Nodes vaoCircle/vbo
        glGenVertexArrays(2, vaoCircle);
        glBindVertexArray(vaoCircle[0]);

        glGenBuffers(2, vboCircle);
        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * nv, vertices, GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindVertexArray(vaoCircle[1]);

        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * nv, verticesOut, GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    vec2 get2d() {
        return vec2(pos.x, pos.y);
    }

    bool diffrent(vec3 c1, vec3 c2) {
        return color[0] != c1 || color[0] != c2 || color[1] != c1 || color[1] != c2;
    }

    void Draw(mat4 TVPmat) {
        gpuProgram.setUniform(color[0], "color");
        glBindVertexArray(vaoCircle[0]);
        glDrawArrays(GL_TRIANGLE_FAN, 0 , nv);

        gpuProgram.setUniform(color[1], "color");
        glBindVertexArray(vaoCircle[1]);
        glDrawArrays(GL_TRIANGLE_FAN, 0 , nv);
    }
};

class Graph {
    unsigned int vaoEdges, vboEdges;
    std::vector<Node> nodes;
    std::vector<vec4> edges;

    void generateNodes() {
        vec2 wSize = camera.getSize();
        while (nodes.size() < 50) {
            float randX = rand() % ((int) wSize.x - 20) - (int) wSize.x / 2 + 10;
            float randY = rand() % ((int) wSize.y - 20) - (int) wSize.y / 2 + 10;

            bool near = false;

            for (int i = 0; i < nodes.size(); ++i) {
                vec2 tmp = nodes[i].get2d();
                if (sqrt(pow(tmp.x - randX, 2) + pow(tmp.y - randY, 2)) < 20) {
                    near = true;
                    continue;
                }
            }
            vec3 color1;
            vec3 color2;
            bool identical = true;
            while (identical) {
                identical = false;
                color1 = vec3((float) (rand() % 100) / 100.0f, (float) (rand() % 100) / 100.0f,
                                   (float) (rand() % 100) / 100.0f);
                color2 = vec3((float) (rand() % 100) / 100.0f, (float) (rand() % 100) / 100.0f,
                                   (float) (rand() % 100) / 100.0f);
                for (auto i = nodes.begin(); i != nodes.end(); i++) {
                    if (!(*i).diffrent(color1, color2))
                        identical = true;
                }
            }
            if (!near && color1 != color2)
                nodes.emplace_back(Node(vec3(randX, randY, sqrt(pow(randX, 2) + pow(randY, 2) + 1)), color1, color2));
        }
    }

    void generateEdges() {
        int maxEdgenum = (50 * 49) / 2;
        std::vector<vec2> pairs;
        while ((float)edges.size() / (float)maxEdgenum < 0.05) {
            int n1 = rand() % 50;
            int n2 = rand() % 50;

            vec2 n1Pos = nodes[n1].get2d();
            vec2 n2Pos = nodes[n2].get2d();

            if (n1 != n2) {
                bool has = false;
                for (auto iter = pairs.begin(); iter != pairs.end(); iter++) {
                    vec2 tmp = *iter;

                    if (tmp == vec2(n1, n2) || tmp == vec2(n2, n1)) {
                        has = true;
                    }
                }
                if (!has) {
                    edges.emplace_back(n1Pos.x, n1Pos.y, n2Pos.x, n2Pos.y);
                    pairs.emplace_back(n1, n2);
                }
            }
        }
        printf("%d\n", edges.size());
        for (int i = 0; i < edges.size(); ++i)
            printf("x1: %f y1: %f x2: %f y2: %f\n", edges[i].x, edges[i].y, edges[i].z, edges[i].w);
    }
public:
    void Create() {
        //Generate Nodes location.
        generateNodes();

        //Generate edges.
        generateEdges();

        //Make Edge vaoCircle/vbo
        glGenVertexArrays(1, &vaoEdges);
        glBindVertexArray(vaoEdges);

        glGenBuffers(1, &vboEdges);
        glBindBuffer(GL_ARRAY_BUFFER, vboEdges);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec4) * edges.size(), &edges[0], GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        mat4 VPTransform = camera.V() * camera.P();
        gpuProgram.setUniform(VPTransform, "MVP");

        //draw Edges
        gpuProgram.setUniform(vec3(1,1,0), "color");
        glLineWidth(2.0f);
        glBindVertexArray(vaoEdges);
        glDrawArrays(GL_LINES, 0 , edges.size() * 2);

        //draw Nodes
        for (auto i = nodes.begin(); i != nodes.end(); i++)
            (*i).Draw(VPTransform);
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
