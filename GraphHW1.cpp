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
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec3 vc;

    out vec3 getColor;

	void main() {
        getColor = vc;
		gl_Position = vec4(vp.x, vp.y, 0, vp.z) * MVP;		// transform vp from modeling space to normalized device space
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

GPUProgram gpuProgram;
bool operator!=(vec3 a, vec3 b) {
    return (a.x != b.x and a.y != b.y and a.z != b.z);
}

bool operator==(vec3 a, vec3 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z);
}

bool operator==(vec4 a, vec4 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z and a.w == b.w);
}

vec3 toHyperbola(vec2 pos) {
    float d = sqrtf(powf(pos.x, 2) + powf(pos.y, 2));
    vec2 v = {pos.x / d, pos.y / d};
    return {v.x * sinhf(d), v.y * sinhf(d), coshf(d)};
}

int nv = 100;

class Node {
    unsigned int vaoCircle[2]{}, vboCircle[2]{};
    vec2 pos;
    vec3 color[2];

public:
    Node(vec2 pos, vec3 color1, vec3 color2) : pos(pos) {
        color[0] = color1;
        color[1] = color2;
        //Generate circle vertexes.
        vec3 vertices[nv];
        vec3 verticesOut[nv];

        for (int i = 0; i < nv; ++i) {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = toHyperbola({pos.x + 0.06f * cosf(fi), pos.y + 0.06f * sinf(fi)});
            verticesOut[i] = toHyperbola({pos.x + 0.03f * cosf(fi), pos.y + 0.03f * sinf(fi)});
        }

        //Make Nodes vaoCircle/vbo
        glGenVertexArrays(2, vaoCircle);
        glBindVertexArray(vaoCircle[0]);

        glGenBuffers(2, vboCircle);

        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, vertices, GL_STATIC_DRAW);// we do not change later
        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindVertexArray(vaoCircle[1]);

        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, verticesOut, GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    vec2 get2d() {
        return {pos.x, pos.y};
    }

    vec3 hyperPos() {
        return toHyperbola({pos.x, pos.y});
    }

    bool different(vec3 c1, vec3 c2) {
        return color[0] != c1 || color[0] != c2 || color[1] != c1 || color[1] != c2;
    }

    bool different(vec3 c1, vec3 c2, float avg) {
        return (length(c1 - color[0]) < avg and length(c2 - color[1]) < avg);
    }

    void print() {
        printf("x: %f y: %f\n", pos.x, pos.y);
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

struct Edge {
    float x1, y1, w1, x2, y2, w2;

    Edge(float x1, float y1, float w1, float x2, float y2, float w2)
    : x1(x1), y1(y1), w1(w1), x2(x2), y2(y2), w2(w2) { }
};

class Graph {
    unsigned int vaoEdges, vboEdges;
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    void generateNodes() {
        while (nodes.size() < 50) {
            float randX = ((float)(rand() % 300 - 150) / 100.0f);
            float randY = ((float)(rand() % 300 - 150) / 100.0f);

            bool near = false;

            for (auto & node : nodes) {
                vec2 tmp = node.get2d();
                if (length(vec2(randX, randY) - tmp) < 0.15) {
                    near = true;
                    continue;
                }
            }
            vec3 color1;
            vec3 color2;
            bool identical = true;
            while (identical) {
                identical = false;
                color1 = vec3((float) (rand() % 80) / 100.0f + 0.2, (float) (rand() % 80) / 100.0f + 0.2,
                                   (float) (rand() % 80) / 100.0f + 0.2);
                color2 = vec3((float) (rand() % 80) / 100.0f + 0.2, (float) (rand() % 80) / 100.0f + 0.2,
                                   (float) (rand() % 80) / 100.0f + 0.2);
                for (auto & node : nodes) {
                    if (!node.different(color1, color2) or node.different(color1, color2, 0.3))
                        identical = true;
                }
            }
            if (!near and color1 != color2 and length(color2 - color1) > 0.9)
                nodes.emplace_back(Node({randX, randY}, color1, color2));
        }
    }

    void generateEdges() {
        int maxEdgenum = (50 * 49) / 2;
        std::vector<vec2> pairs;
        while ((float)edges.size() / (float)maxEdgenum < 0.05) {
            int n1 = rand() % 50;
            int n2 = rand() % 50;

            vec3 n1Pos = nodes[n1].hyperPos();
            vec3 n2Pos = nodes[n2].hyperPos();

            if (n1 != n2) {
                bool has = false;
                for (auto tmp : pairs) {
                    if (tmp == vec2(n1, n2) || tmp == vec2(n2, n1)) {
                        has = true;
                    }
                }
                if (!has) {

                    edges.emplace_back(Edge(n1Pos.x, n1Pos.y, n1Pos.z, n2Pos.x, n2Pos.y, n2Pos.z));
                    pairs.emplace_back(n1, n2);
                }
            }
        }
    }

public:
    void Create() {
        //Generate Nodes location.
        generateNodes();

        for (auto & node : nodes)
            node.print();

        //Generate edges.
        generateEdges();

        //Make Edge vaoCircle/vbo
        glGenVertexArrays(1, &vaoEdges);
        glBindVertexArray(vaoEdges);

        glGenBuffers(1, &vboEdges);
        glBindBuffer(GL_ARRAY_BUFFER, vboEdges);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Edge) * edges.size(), &edges[0], GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        mat4 VPTransform = {1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1};
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
